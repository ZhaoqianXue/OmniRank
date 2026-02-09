"""Security tests for upload path traversal protections."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_upload_rejects_traversal_filename():
    client = TestClient(app)
    csv_content = "a,b\n1,2\n"

    response = client.post(
        "/api/upload",
        files={"file": ("../evil.csv", csv_content, "text/csv")},
    )

    assert response.status_code == 400
    assert "Unsafe filename path" in response.json().get("detail", "")
