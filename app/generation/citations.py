import re


class CitationExtractor:

    CITATION_PATTERN = re.compile(
        r"\[(\d+)\]"
    )

    def extract(self, answer: str) -> list[int]:

        if not answer.strip():
            return []

        citations = self.CITATION_PATTERN.findall(
            answer
        )

        return sorted(
            set(int(citation) for citation in citations)
        )
