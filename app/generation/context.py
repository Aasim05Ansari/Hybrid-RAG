from app.retrieval.models import RetrievalResult


class ContextBuilder:
    """
    Converts retrieved chunks into numbered context
    that can be supplied to a grounded LLM prompt.
    """

    def build(self, results: list[RetrievalResult]) -> str:
        if not results:
            return ""

        context_parts = []

        for index, result in enumerate(results, start=1):
            source = result.metadata.get("source", "unknown")
            section = result.metadata.get("section", "unknown")

            context_parts.append(
                f"[{index}]\n"
                f"Source: {source}\n"
                f"Section: {section}\n"
                f"Content:\n"
                f"{result.content}"
            )

        return "\n\n".join(context_parts)
