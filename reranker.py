import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, docs: list[str] = None) -> None:
        self.docs = docs
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def set_docs(self, docs: list[str]) -> None:
        self.docs = docs

    def get_docs(self, question: str, n: int = 3) -> list[str]:
        pairs = [(question, doc) for doc in self.docs]
        scores = self.cross_encoder.predict(pairs)

        sorted_indices = np.argsort(scores)
        reranker_docs = [self.docs[i] for i in sorted_indices[-n:]]
        return reranker_docs
