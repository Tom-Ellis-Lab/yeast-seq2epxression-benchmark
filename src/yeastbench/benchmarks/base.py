from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

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
