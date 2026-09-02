class GroundedPromptBuilder:
    """
    Builds a prompt that instructs the LLM to answer
    strictly from retrieved context.
    """

    SYSTEM_INSTRUCTIONS = """You are a grounded question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume facts that are not present in the context.
3. If the context does not contain enough information to answer the question, say:
   "I don't have enough information in the provided documents to answer this question."
4. When making a claim based on the context, cite the relevant context number using [1], [2], etc.
5. Do not create citations for information that is not supported by the context.
"""

    def build(self, query: str, context: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not context.strip():
            raise ValueError("Context cannot be empty")

        return (
            f"{self.SYSTEM_INSTRUCTIONS}\n\n"
            f"CONTEXT:\n"
            f"{context}\n\n"
            f"USER QUESTION:\n"
            f"{query.strip()}"
        )
