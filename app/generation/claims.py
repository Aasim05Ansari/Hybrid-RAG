from dataclasses import dataclass


@dataclass
class AnswerClaim:
    claim: str
    citation: int | None = None


class ClaimExtractor:

    def extract(self, answer: str) -> list[AnswerClaim]:

        if not answer.strip():
            return []

        claims = []

        for sentence in self._split_sentences(answer):
            sentence = sentence.strip()

            if not sentence:
                continue

            citations = self._extract_citations(sentence)

            if citations:
                claim_text = self._remove_citations(sentence)

                for citation in citations:
                    claims.append(
                        AnswerClaim(
                            claim=claim_text.strip(),
                            citation=citation,
                        )
                    )
            else:
                claims.append(
                    AnswerClaim(
                        claim=sentence,
                        citation=None,
                    )
                )

        return claims

    @staticmethod
    def _split_sentences(answer: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in answer.split(".")
            if sentence.strip()
        ]

    @staticmethod
    def _extract_citations(sentence: str) -> list[int]:
        import re

        return [
            int(value)
            for value in re.findall(
                r"\[(\d+)\]",
                sentence,
            )
        ]

    @staticmethod
    def _remove_citations(sentence: str) -> str:
        import re

        return re.sub(
            r"\s*\[\d+\]",
            "",
            sentence,
        ).strip()
