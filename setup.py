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
        "faiss-cpu==1.13.2",
        "langchain-community==0.3.31",
        "langchain-openai==1.1.7",
        "langchain-text-splitters==1.1.0",
        "python-dotenv==1.2.1",
        "youtube-transcript-api==1.2.3",
    ],
)
