import os

from langchain_community.vectorstores import FAISS


class FAISS_Helper:
    DEACTIVATE_CACHE = True
    CACHE_DIR = ".faiss_cache"

    def cache_key_from_url(self, url):
        key = url.replace("/", "_").replace("?", "_")
        # remove all non-alphanumeric characters
        key = "".join([c for c in key if c.isalnum() or c == "_"])
        return key

    def filename(self, cache_key):
        return f"{FAISS_Helper.CACHE_DIR}/{cache_key}.bin"

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def save_to_cache(self, cache_key, db):
        if self.DEACTIVATE_CACHE:
            return
        self.mkdir(FAISS_Helper.CACHE_DIR)
        filename = self.filename(cache_key)
        db.save_local(filename)

    def load_from_cache(self, cache_key, embeddings):
        if self.DEACTIVATE_CACHE:
            return None
        filename = self.filename(cache_key)
        if os.path.exists(filename):
            return FAISS.load_local(filename, embeddings)
        return None
