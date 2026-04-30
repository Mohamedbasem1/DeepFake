from pathlib import Path

from deepfake_detector.config import parse_config
from deepfake_detector.predict import Prediction, make_item_id
from deepfake_detector.submission import write_predictions


def test_write_predictions_uses_configured_columns(tmp_path: Path) -> None:
    config = parse_config(
        {
            "submission": {
                "columns": {"id": "filename", "score": "prob", "label": "class"},
                "labels": {"real": "0", "fake": "1"},
            }
        }
    )
    output = tmp_path / "submission.csv"
    predictions = [Prediction("sample", tmp_path / "sample.jpg", 0.75, "1")]

    write_predictions(output, predictions, config.submission)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "filename,prob,class",
        "sample,0.750000,1",
    ]


def test_make_item_id_modes(tmp_path: Path) -> None:
    root = tmp_path / "data"
    nested = root / "a" / "image.fake.jpg"
    nested.parent.mkdir(parents=True)
    nested.touch()

    assert make_item_id(nested, root, "stem") == "image.fake"
    assert make_item_id(nested, root, "name") == "image.fake.jpg"
    assert make_item_id(nested, root, "relative") == "a/image.fake.jpg"


def test_write_predictions_can_match_official_submission_columns(tmp_path: Path) -> None:
    config = parse_config(
        {
            "submission": {
                "id_from": "name",
                "columns": {"id": "full_secret_name", "score": None, "label": "prediction"},
                "labels": {"real": 0, "fake": 1},
            }
        }
    )
    output = tmp_path / "submission.csv"
    predictions = [Prediction("abc.png", tmp_path / "abc.png", 0.75, "1")]

    write_predictions(output, predictions, config.submission)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "full_secret_name,prediction",
        "abc.png,1",
    ]
