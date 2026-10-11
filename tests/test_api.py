import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "deployment")))
import main  # noqa: E402


class FakeModel:
    names = {0: "TANK"}

    def predict(self, source, conf, verbose=False):
        box = SimpleNamespace(
            cls=[0],
            conf=[0.9],
            xyxy=[SimpleNamespace(tolist=lambda: [10.0, 20.0, 50.0, 60.0])],
        )
        return [SimpleNamespace(boxes=[box])]


@pytest.fixture
def client():
    return TestClient(main.app)


def _jpeg_bytes() -> bytes:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "gpu_available" in body


def test_predict_rejects_non_image(client):
    files = {"file": ("test.txt", b"this is a test", "text/plain")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 400
    assert "not an image" in response.json()["detail"].lower()


def test_predict_rejects_corrupt_image(client):
    files = {"file": ("bad.jpg", b"not really a jpeg", "image/jpeg")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 400


def test_predict_without_weights_returns_503(client, monkeypatch):
    monkeypatch.setattr(main, "MODEL_PATH", "does/not/exist.pt")
    main.get_model.cache_clear()
    files = {"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 503


def test_predict_returns_detections(client, monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    files = {"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")}
    response = client.post("/predict/", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["predictions"] == [
        {
            "class_id": 0,
            "class_name": "TANK",
            "confidence": 0.9,
            "bbox": [10.0, 20.0, 50.0, 60.0],
        }
    ]
