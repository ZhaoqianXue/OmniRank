"""Security tests for report markdown sanitization."""

from __future__ import annotations

from core.schemas import RankingResults
from tools.generate_report import generate_report


def test_generate_report_escapes_untrusted_content():
    results = RankingResults(
        items=["<script>alert('x')</script>", "B"],
        theta_hat=[0.8, 0.2],
        ranks=[1, 2],
        ci_lower=[1.0, 2.0],
        ci_upper=[2.0, 3.0],
        indicator_value=None,
    )

    report = generate_report(
        results=results,
        session_meta={
            "B": 2000,
            "seed": 42,
            "current_file_path": "/tmp/input_<img src=x onerror=alert(1)>.csv",
            "r_script_path": "src/spectral_ranking/spectral_ranking.R",
        },
        plots=[],
    )

    assert "<script>" not in report.markdown
    assert "&lt;script&gt;" in report.markdown
    assert "onerror=alert" in report.markdown  # text is preserved as escaped content
