from litellm import completion

from retriever import Retriever
from reranker import Reranker


class QuestionAnsweringBot:
    def __init__(self, docs, method) -> None:
        self.retriever = Retriever(docs, method)
        self.reranker = Reranker()
        self.retriever_documents = None
        self.reranker_documents = None

    def answer_question(self, question: str) -> str:
        self.retriever_documents = self.retriever.get_docs(question, n=50)
        self.reranker.set_docs(self.retriever_documents)
        self.reranker_documents = self.reranker.get_docs(question)

        document_references = [f"[{i + 1}]" for i in range(len(self.reranker_documents))]
        context_with_references = "\n".join(
            [f"{doc} {document_references[i]}" for i, doc in enumerate(self.reranker_documents)]
        )
        response = completion(
            model="gpt-4o-mini",
            messages=[{"content": f"You are a question answering bot, please answer "
                                  f"to the *user question* based on *provided context* and include citations"
                                  f"(it's in the end of each context in brackets, like [1]."
                                  f"*user question*: ###{question}###"
                                  f"*provided context*: ###{context_with_references}###",
                       "role": "user"}]
        )

        self.reranker_documents = context_with_references
        return response.choices[0].message.content
