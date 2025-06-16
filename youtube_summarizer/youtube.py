"""
GPT YouTube Summarizer

Find OpenAI API Keys:  https://platform.openai.com/account/api-keys  (in OpenAI not ChatGPT)
then...
export OPENAI_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

FAISS tips - https://github.com/matsui528/faiss_tips
"""

import argparse
import os
import sys

from dotenv import find_dotenv, load_dotenv
from youtube_summarizer.faiss_helper import FAISS_Helper
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def create_db_from_youtube_video_url(video_url: str, embeddings) -> FAISS:
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

        Your answers should be in Markdown format, especially for lists and tables.
        """
    return template


def get_response_from_query_chatgpt(db, query, k=4):
    """
    gpt-3.5-turbo can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    model_name = "o1"
    chat = ChatOpenAI(model_name=model_name)

    # Template to use for the system message prompt
    template = get_template_string()

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | chat

    response = chain.invoke({"question": query, "docs": docs_page_content})
    response = response.content
    return response, docs


def youtube_summarizer():
    parser = argparse.ArgumentParser(description="Process some internet data.")
    parser.add_argument(
        "-u", "--url", type=str, required=False, help="The YouTube video URL to process"
    )
    parser.add_argument("-q", "--query", type=str, required=False, help="Your question")
    args = parser.parse_args()

    if args.url:
        video_url = args.url
    else:
        video_url = input("Enter the YouTube video URL: ")

    query = "Summarize the video, and state any conclusions the presenter makes."
    if args.query:
        query = args.query

    # OpenAI
    load_dotenv(find_dotenv())
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Set OPENAI_API_KEY environment variable")
        return 1

    # Sample URL:
    # video_url = "https://www.youtube.com/watch?v=C3yuV8-r8UI"  # 15 free things in Las Vegas
    embeddings = OpenAIEmbeddings()
    db = create_db_from_youtube_video_url(video_url, embeddings)

    print("")
    response, docs = get_response_from_query_chatgpt(db, query)
    print(response)
    print("")

    return 0


if __name__ == "__main__":
    sys.exit(youtube_summarizer())
