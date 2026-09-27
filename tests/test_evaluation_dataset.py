import json

import pytest

from app.evaluation.dataset import EvaluationCase, EvaluationDataset


def test_evaluation_case_creation():
    case = EvaluationCase(
        id="q001",
        question="How many sick leave days can employees take?",
        expected_answer="Employees may take up to 12 days of sick leave.",
        expected_sources=["leave.txt"],
        expected_chunk_ids=["sick-leave"],
    )

    assert case.id == "q001"
    assert case.question == "How many sick leave days can employees take?"
    assert case.expected_answer == (
        "Employees may take up to 12 days of sick leave."
    )
    assert case.expected_sources == ["leave.txt"]
    assert case.expected_chunk_ids == ["sick-leave"]


def test_evaluation_case_rejects_empty_id():
    with pytest.raises(ValueError, match="id cannot be empty"):
        EvaluationCase(
            id="",
            question="What is the leave policy?",
            expected_answer="Employees can take leave.",
        )


def test_evaluation_case_rejects_empty_question():
    with pytest.raises(ValueError, match="question cannot be empty"):
        EvaluationCase(
            id="q001",
            question="",
            expected_answer="Employees can take leave.",
        )


def test_evaluation_case_rejects_empty_expected_answer():
    with pytest.raises(ValueError, match="expected_answer cannot be empty"):
        EvaluationCase(
            id="q001",
            question="What is the leave policy?",
            expected_answer="",
        )


def test_dataset_creation():
    cases = [
        EvaluationCase(
            id="q001",
            question="What is the leave policy?",
            expected_answer="Employees can take leave.",
        ),
        EvaluationCase(
            id="q002",
            question="How many leave days are allowed?",
            expected_answer="Employees may take 12 days.",
        ),
    ]

    dataset = EvaluationDataset(cases)

    assert len(dataset) == 2
    assert dataset.cases == cases


def test_dataset_rejects_empty_cases():
    with pytest.raises(ValueError, match="cannot be empty"):
        EvaluationDataset([])


def test_dataset_is_iterable():
    cases = [
        EvaluationCase(
            id="q001",
            question="What is the leave policy?",
            expected_answer="Employees can take leave.",
        ),
        EvaluationCase(
            id="q002",
            question="How many leave days are allowed?",
            expected_answer="Employees may take 12 days.",
        ),
    ]

    dataset = EvaluationDataset(cases)

    assert list(dataset) == cases


def test_dataset_loads_from_json(tmp_path):
    data = [
        {
            "id": "q001",
            "question": "How many sick leave days can employees take?",
            "expected_answer": "Employees may take up to 12 days.",
            "expected_sources": ["leave.txt"],
            "expected_chunk_ids": ["sick-leave"],
        }
    ]

    dataset_file = tmp_path / "evaluation.json"

    with dataset_file.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    dataset = EvaluationDataset.from_json(dataset_file)

    assert len(dataset) == 1

    case = dataset.cases[0]

    assert case.id == "q001"
    assert case.question == "How many sick leave days can employees take?"
    assert case.expected_answer == "Employees may take up to 12 days."
    assert case.expected_sources == ["leave.txt"]
    assert case.expected_chunk_ids == ["sick-leave"]


def test_dataset_rejects_missing_file(tmp_path):
    missing_file = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        EvaluationDataset.from_json(missing_file)


def test_dataset_rejects_non_list_json(tmp_path):
    dataset_file = tmp_path / "evaluation.json"

    with dataset_file.open("w", encoding="utf-8") as file:
        json.dump({"id": "q001"}, file)

    with pytest.raises(
        ValueError,
        match="JSON must contain a list",
    ):
        EvaluationDataset.from_json(dataset_file)