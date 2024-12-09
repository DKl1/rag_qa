def chunk_document(doc: str, desired_chunk_size: int = 100, max_chunk_size: int = 3000) -> str:
    chunk = ''

    for line in doc.splitlines():
        chunk += line + '\n'

        if len(chunk) >= desired_chunk_size:
            yield chunk[:max_chunk_size]
            chunk = ''

    if chunk:
        yield chunk


def chunk_documents(docs: list[str], desired_chunk_size: int = 100, max_chunk_size: int = 3000) -> list[str]:
    chunks = []
    for doc in docs:
        chunks += list(chunk_document(doc, desired_chunk_size, max_chunk_size))
    return chunks
