"""方法间比较的统计量：配对差异的 bootstrap 置信区间与符号翻转置换检验。

比较单位是**目标**（一个被测文件 / 函数）：同一个目标上，两种方法各跑若干次，先在目标内取均值，
再对“各目标上的差异”做推断。这样：
- 配对消掉了目标难度的差异（functoolz 和 itertoolz 的绝对分数差 30 个点，远大于方法差异）；
- 同一目标的多次运行不当成独立样本（它们共享目标，不独立），推断的样本量是目标数。

只用标准库：样本量小，不值得为此引入 numpy / scipy；随机数固定种子，结果可复现。
"""

from __future__ import annotations

import itertools
import random
import statistics
from dataclasses import dataclass
from typing import Mapping, Sequence

N_RESAMPLES = 10_000
EXACT_PERMUTATION_MAX = 16  # 目标数不超过这个值时枚举全部 2^n 种符号翻转


@dataclass(frozen=True)
class PairedComparison:
    n_targets: int
    mean_a: float
    mean_b: float
    mean_diff: float  # b - a
    ci_low: float
    ci_high: float
    p_value: float  # 双侧
    wins: int  # b > a 的目标数
    losses: int
    ties: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def paired_diffs(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]]) -> list[tuple[float, float]]:
    """两种方法共同的目标上，各自的目标内均值。缺任何一方的目标不参与比较。"""
    common = sorted(set(a) & set(b))
    return [(statistics.fmean(a[t]), statistics.fmean(b[t])) for t in common if a[t] and b[t]]


def bootstrap_ci(values: Sequence[float], level: float = 0.95, n: int = N_RESAMPLES, seed: int = 0) -> tuple[float, float]:
    """均值的百分位 bootstrap 置信区间。"""
    if not values:
        raise ValueError("empty sample")
    rng = random.Random(seed)
    k = len(values)
    means = sorted(statistics.fmean(rng.choices(values, k=k)) for _ in range(n))
    alpha = (1 - level) / 2
    return means[int(alpha * n)], means[min(n - 1, int((1 - alpha) * n))]


def sign_flip_p_value(diffs: Sequence[float], n: int = N_RESAMPLES, seed: int = 0) -> float:
    """配对符号翻转置换检验（H0：差异关于 0 对称），双侧 p 值。

    目标数少时精确枚举；否则蒙特卡洛，p 值按 (count + 1) / (n + 1) 计，不会报出 0。
    """
    if not diffs:
        raise ValueError("empty sample")
    observed = abs(sum(diffs))
    eps = 1e-12 * max(1.0, observed)
    if len(diffs) <= EXACT_PERMUTATION_MAX:
        total = count = 0
        for signs in itertools.product((1, -1), repeat=len(diffs)):
            total += 1
            count += abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - eps
        return count / total
    rng = random.Random(seed)
    count = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - eps for _ in range(n))
    return (count + 1) / (n + 1)


def compare(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]], level: float = 0.95) -> PairedComparison:
    """``a`` / ``b``：目标 -> 该方法在这个目标上各次运行的得分。返回 b 相对 a 的配对比较。"""
    pairs = paired_diffs(a, b)
    if not pairs:
        raise ValueError("no common targets")
    diffs = [y - x for x, y in pairs]
    low, high = bootstrap_ci(diffs, level)
    return PairedComparison(
        n_targets=len(pairs),
        mean_a=statistics.fmean(x for x, _ in pairs),
        mean_b=statistics.fmean(y for _, y in pairs),
        mean_diff=statistics.fmean(diffs),
        ci_low=low,
        ci_high=high,
        p_value=sign_flip_p_value(diffs),
        wins=sum(d > 0 for d in diffs),
        losses=sum(d < 0 for d in diffs),
        ties=sum(d == 0 for d in diffs),
    )
