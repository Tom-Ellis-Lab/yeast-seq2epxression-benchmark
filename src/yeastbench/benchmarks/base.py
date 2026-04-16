from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

AdapterT = TypeVar("AdapterT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class BenchmarkInfo:
    name: str
    version: str
    description: str
    distribution_uri: str


class Benchmark(ABC, Generic[AdapterT, ResultT]):
    info: BenchmarkInfo
    adapter_protocol: ClassVar[type]

    @abstractmethod
    def evaluate(self, adapter: AdapterT) -> ResultT: ...

    @abstractmethod
    def plot(self, results: ResultT, out_dir: Path) -> None: ...

    @abstractmethod
    def save_results(self, results: ResultT, out_dir: Path) -> None:
        """Persist raw results (scores, labels, metadata) to *out_dir*."""
        ...

    @abstractmethod
    def load_results(self, out_dir: Path) -> ResultT:
        """Reconstruct results from files previously written by *save_results*."""
        ...

    @abstractmethod
    def summary_dict(self, results: ResultT) -> dict[str, Any]:
        """Return a JSON-serialisable summary of *results*."""
        ...

    @abstractmethod
    def headline(self, results: ResultT) -> str:
        """One-line summary for CLI output after evaluation."""
        ...
