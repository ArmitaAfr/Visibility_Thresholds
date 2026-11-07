"""CPU-only comparison of existing judge exports; no inference dependencies."""

from .compare import analyze_exports, write_report

__all__ = ["analyze_exports", "write_report"]
