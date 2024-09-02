"""
GPT YouTube Summarizer

Find OpenAI API Keys:  https://platform.openai.com/account/api-keys  (in OpenAI not ChatGPT)
then...
export OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
"""

import argparse
import os
import sys
import textwrap

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from faiss_helper import FAISS_Helper


# FAISS tips - https://github.com/matsui528/faiss_tips


def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    helper = FAISS_Helper()
    cache_key = helper.cache_key_from_url(video_url)
    print(f"cache_key: {cache_key}")
    db = helper.load_from_cache(cache_key, embeddings)
    if db:
        print("loaded from cache")
        return db

    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    print("saving to cache")
    helper.save_to_cache(cache_key, db)
    return db


def get_template_string():
    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can answer questions about youtube videos
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.
        Do not include any advertisements or promotions in your answer.
        Do not report any comments about YouTube likes, subscribe or notifications.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
        """
    return template


def get_response_from_query_chatgpt(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)

    # Template to use for the system message prompt
    template = get_template_string()

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def get_response_from_query_davinci(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name="text-davinci-003")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=get_template_string(),
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


parser = argparse.ArgumentParser(description="Process some internet data.")
parser.add_argument(
    "-u", "--url", type=str, required=False, help="The YouTube video URL to process"
)
parser.add_argument("-q", "--query", type=str, required=False, help="Your question")
parser.add_argument(
    "-c", "--chatgpt", type=str, required=False, help="Use ChatGPT 4 model (default)"
)
parser.add_argument(
    "-d", "--davinci", type=str, required=False, help="Use DaVinci model"
)
args = parser.parse_args()

if args.url:
    video_url = args.url
else:
    video_url = input("Enter the YouTube video URL: ")

query = "Summarize the video, and state any conclusions the presenter makes."
if args.query:
    query = args.query

model = "chatgpt"
if args.davinci:
    model = "davinci"

# OpenAI
load_dotenv(find_dotenv())
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print("Set OPENAI_API_KEY environment variable")
    sys.exit(1)

embeddings = OpenAIEmbeddings()

# Example usage:
# video_url = "https://www.youtube.com/watch?v=C3yuV8-r8UI"  # 15 free things in Las Vegas
db = create_db_from_youtube_video_url(video_url)

print("")
response = ""
if model == "chatgpt":
    response, docs = get_response_from_query_chatgpt(db, query)
elif model == "davinci":
    response, docs = get_response_from_query_davinci(db, query)

print(textwrap.fill(response, width=85))
sys.exit(0)
