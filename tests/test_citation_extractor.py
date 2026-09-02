from app.generation.citations import CitationExtractor


def test_extracts_citations():
    extractor = CitationExtractor()

    citations = extractor.extract(
        "Employees receive 24 days of annual leave [1]."
    )

    assert citations == [1]


def test_extracts_multiple_citations():
    extractor = CitationExtractor()

    citations = extractor.extract(
        "Annual leave is 24 days [1]. "
        "Sick leave is 12 days [2]."
    )

    assert citations == [1, 2]


def test_removes_duplicate_citations():
    extractor = CitationExtractor()

    citations = extractor.extract(
        "Annual leave is 24 days [1]. "
        "The policy confirms this [1]."
    )

    assert citations == [1]


def test_returns_sorted_citations():
    extractor = CitationExtractor()

    citations = extractor.extract(
        "See [3], [1], and [2]."
    )

    assert citations == [1, 2, 3]


def test_returns_empty_for_answer_without_citations():
    extractor = CitationExtractor()

    citations = extractor.extract(
        "I don't have enough information."
    )

    assert citations == []


def test_returns_empty_for_empty_answer():
    extractor = CitationExtractor()

    citations = extractor.extract("")

    assert citations == []
