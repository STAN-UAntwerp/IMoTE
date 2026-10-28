from typing import Iterable, List, Optional, Tuple


class LabelFormatter:
    """Formats the parts of node labels: feature names, numbers and linear expressions.

    Attributes:
        feature_names: Name to show for every feature, or None to use X0, X1, ...
    """

    feature_names: Optional[List[str]]

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Args:
            feature_names: Name to show for every feature, or None to use X0, X1, ...
        """
        self.feature_names = feature_names

    def feature(self, idx: int) -> str:
        """Returns the name to show for feature idx."""
        return f"X{idx}" if self.feature_names is None else self.feature_names[idx]

    @staticmethod
    def number(value: float) -> str:
        """Returns a number with 3 significant digits."""
        return f"{value:.3g}"

    def expression(self, terms: Iterable[Tuple[int, float]], intercept: float) -> str:
        """Returns a linear expression like "0.35·x - 0.1·y + 2", leaving out zero terms and a zero intercept.

        Args:
            terms: (feature index, coefficient) of every term.
            intercept: Constant added to the terms.
        """
        parts = [f"{self.number(coef)}·{self.feature(idx)}" for idx, coef in terms if coef != 0]
        if intercept != 0 or not parts:
            parts.append(self.number(intercept))
        return " + ".join(parts).replace("+ -", "- ")
