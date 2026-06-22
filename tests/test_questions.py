import json
from pathlib import Path


def test_evaluation_questions_have_required_fields():
    questions = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))

    assert questions
    for item in questions:
        assert item["question"]
        assert item["expected"]
        assert item["intent"] in {"rag", "sql", "mixed", "visualization"}

