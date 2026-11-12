from __future__ import annotations

import numpy as np

REGRESSION_NODES = ["con", "lin", "pcon", "plin"]
NODE_PREFERENCE_ORDER = REGRESSION_NODES + ["pconc"]

class Split:
    def __init__(
        self,
        node: str = "",
        feature: int = -1,
        pivot: float = float("nan"),
        slope_left: float = 0.0,
        intercept_left: float = 0.0,
        slope_right: float = 0.0,
        intercept_right: float = 0.0,
        interval: np.ndarray = np.array([-np.inf, np.inf]),
        pivot_categorical: list = [0],
        wrss: float = float("inf"),
        bic: float = float("inf"),
        ) -> None:
        
        self.node = node
        self.feature = feature
        self.pivot = pivot
        self.slope_left = slope_left
        self.intercept_left = intercept_left
        self.slope_right = slope_right
        self.intercept_right = intercept_right
        self.interval = interval
        self.pivot_categorical = pivot_categorical
        self.wrss = wrss
        self.bic = bic
    
    def is_improved(self, new_split: Split) -> bool:
        if new_split.bic == self.bic:
            if self.node == "":
                return False
            new_node = new_split.node
            node = self.node
            if new_node == "tlin":
                new_node = "lin"
            if node == "tlin":
                node = "lin"
            return NODE_PREFERENCE_ORDER.index(new_node) < NODE_PREFERENCE_ORDER.index(node)
        return new_split.bic < self.bic