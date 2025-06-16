import os

from langchain_community.vectorstores import FAISS
import os
import tempfile


class FAISS_Helper:
    DEACTIVATE_CACHE = False
    CACHE_DIR = ".faiss_cache"

    def cache_key_from_url(self, url):
        key = url.replace("/", "_").replace("?", "_")
        # remove all non-alphanumeric characters
        key = "".join([c for c in key if c.isalnum() or c == "_"])
        return key

    def cache_dir(self):
        return os.path.join(tempfile.gettempdir(), FAISS_Helper.CACHE_DIR)

    def filename(self, cache_key):
        return os.path.join(self.cache_dir(), f"{cache_key}.bin")

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def save_to_cache(self, cache_key, db):
        if self.DEACTIVATE_CACHE:
            return
        self.mkdir(self.cache_dir())
        filename = self.filename(cache_key)
        db.save_local(filename)

    def load_from_cache(self, cache_key, embeddings):
        if self.DEACTIVATE_CACHE:
            return None
        filename = self.filename(cache_key)
        if os.path.exists(filename):
            print(f"Loading from cace: {filename}")
            return FAISS.load_local(
                filename, embeddings, allow_dangerous_deserialization=True
            )
        return None
