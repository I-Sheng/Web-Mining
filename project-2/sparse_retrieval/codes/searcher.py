import numpy as np
from pyserini.index.lucene import LuceneIndexReader
from pyserini.search.lucene import LuceneSearcher
from jnius import autoclass

class bm25_searcher:
    def __init__(self, index_path: str):
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=2, b=0.75)

    def search(self, query, k=1000):
        return self.searcher.search(query, k = k)[:min(1000, k)]

class laplace_language_model:
    def __init__(self, index_path: str , lambda_param: float = 0.5):
        self.searcher = LuceneSearcher(index_path)
        LMDirichletSimilarity = autoclass("org.apache.lucene.search.similarities.LMDirichletSimilarity")
        self.searcher = LuceneSearcher(index_path)
        self.searcher.object.similarity = LMDirichletSimilarity(lambda_param)

    def search(self, query: str, k: int = 1000):
        return self.searcher.search(query, k=k)

class jelinek_merver_language_model:
    def __init__(self, index_path: str , lambda_param: float = 0.5):
        self.searcher = LuceneSearcher(index_path)
        LMJelinekMercerSimilarity = autoclass("org.apache.lucene.search.similarities.LMJelinekMercerSimilarity")
        self.searcher = LuceneSearcher(index_path)
        self.searcher.object.similarity = LMJelinekMercerSimilarity(lambda_param)

    def search(self, query: str, k: int = 1000):
        return self.searcher.search(query, k=k)



