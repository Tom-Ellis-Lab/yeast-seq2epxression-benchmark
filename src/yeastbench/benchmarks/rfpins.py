from typing import ClassVar
from yeastbench.benchmarks.base import Benchmark
from yeastbench.adapters.protocols import CasetteExpressionPredictor


@dataclass(frozen=True)
class FluorescenceResult:
    name: str


class RFPInsertionBenchmark(Benchmark[CasetteExpressionPredictor, FluorescenceResult]):
    adapter_protocol: ClassVar[type] = CasetteExpressionPredictor

    def __init__(self, cassette_seq, labels_path) -> None:
        self.cassette_seq = cassette_seq
        self.labels_path = labels_path
        pass

    def evaluate(self, adapter: CasetteExpressionPredictor) -> None:
        """
        tbd
        """

        scores = adapter.predict_expressions()

        pass

    def plot(self, results, out_dir) -> None:
        pass
