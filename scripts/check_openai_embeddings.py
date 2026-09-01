from app.embeddings.embedder import OpenAIEmbeddingProvider


def main():

    provider = OpenAIEmbeddingProvider()

    documents = [
        "Annual leave is 24 days per year.",
        "Sick leave is 12 days per year.",
    ]

    document_vectors = provider.embed_documents(documents)

    print("Number of vectors:", len(document_vectors))
    print("Vector dimension:", len(document_vectors[0]))
    print("First 5 values:", document_vectors[0][:5])

    query = "How many days of annual leave are allowed?"

    query_vector = provider.embed_query(query)

    print("Query vector dimension:", len(query_vector))
    print("Query first 5 values:", query_vector[:5])


if __name__ == "__main__":
    main()