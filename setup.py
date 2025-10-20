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
        "langchain-community==0.4",
        "langchain-openai==0.3.28",
        "python-dotenv==1.1.1",
        "youtube-transcript-api==1.1.1",
    ],
)
