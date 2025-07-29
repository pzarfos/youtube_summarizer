from setuptools import setup, find_packages

setup(
    name="youtube_summarizer",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "youtube_summarizer=youtube_summarizer.youtube:youtube_summarizer",
        ],
    },
    install_requires=[
        "faiss-cpu==1.10.0",
        "langchain==0.3.26",
        "langchain-community==0.3.26",
        "langchain-core==0.3.69",
        "langchain-openai==0.3.28",
        "openai==1.97.0",
        "python-dotenv==1.1.1",
        "tiktoken==0.9.0",
        "youtube-transcript-api==1.2.1",
    ],
)
