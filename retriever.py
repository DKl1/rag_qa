import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunker import chunk_documents


class Retriever:
    def __init__(self, docs: list[str] = None, method: str = None) -> None:
        self.docs = chunk_documents(docs, desired_chunk_size=300)
        tokenized_docs = [doc.lower().split(' ') for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.sbert = SentenceTransformer("all-mpnet-base-v2")
        self.doc_embeddings = self.sbert.encode(self.docs)
        self.method = method

    def get_docs(self, question, n=3) -> list[str]:
        print(f"Method: {self.method}")
        if self.method == "Keyword Search (BM25)":
            scores = self._get_bm25_scores(question)
        elif self.method == "Semantic Search":
            scores = self._get_semantic_scored(question)
        else:  # Hybrid approach
            bm25_score = self._get_bm25_scores(question)
            semantic_score = self._get_semantic_scored(question)

            scores = 0.3 * bm25_score + 0.7 * semantic_score

        sorted_indices = np.argsort(scores)
        result = [self.docs[i] for i in sorted_indices[-n:]]
        return result

    def _get_bm25_scores(self, question):
        tokenized_question = question.lower().split(' ')
        scores = self.bm25.get_scores(tokenized_question)
        return torch.tensor(scores)

    def _get_semantic_scored(self, question):
        question_embeddings = self.sbert.encode(question)
        scores = self.sbert.similarity(question_embeddings, self.doc_embeddings)
        return scores[0]
