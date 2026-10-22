"""Educational ID3 (information gain) and C4.5-style (gain ratio) tree learners."""
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def split_information(groups):
    sizes = np.array([len(g) for g in groups], dtype=float)
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return 0.0
    p = sizes / sizes.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


class _GainTree:
    def __init__(self, criterion="gain", max_depth=8, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.feature_names = None
        self.n_features_ = 0

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(str)
        self.feature_names = list(feature_names or [f"feature_{i}" for i in range(X.shape[1])])
        self.n_features_ = X.shape[1]
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        majority = Counter(y).most_common(1)[0][0]
        node = {"prediction": majority, "samples": len(y), "counts": dict(Counter(y))}
        if (len(set(y)) <= 1 or depth >= self.max_depth or
                len(y) < 2 * self.min_samples_leaf or X.shape[1] == 0):
            node["leaf"] = True
            return node

        base = entropy(y)
        best_score, best_feature, best_threshold, best_mask = -1.0, None, None, None
        n_samples, n_features = X.shape

        for j in range(n_features):
            values = np.unique(X[:, j])
            if len(values) < 2:
                continue
            # Candidate thresholds are midpoints between sorted unique values.
            if len(values) > 32:
                values = np.quantile(values, np.linspace(0, 1, 32))
                values = np.unique(values)
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                mask = X[:, j] <= threshold
                left_n, right_n = int(mask.sum()), int((~mask).sum())
                if min(left_n, right_n) < self.min_samples_leaf:
                    continue
                left_y, right_y = y[mask], y[~mask]
                weighted_child_entropy = (left_n / n_samples) * entropy(left_y) + (right_n / n_samples) * entropy(right_y)
                gain = base - weighted_child_entropy
                if self.criterion == "gain_ratio":
                    si = split_information([left_y, right_y])
                    score = gain / si if si > 1e-12 else 0.0
                else:
                    score = gain
                if score > best_score + 1e-12:
                    best_score, best_feature, best_threshold, best_mask = score, j, float(threshold), mask

        if best_feature is None or best_score <= 1e-12:
            node["leaf"] = True
            return node

        node.update({"leaf": False, "feature": best_feature,
                     "feature_name": self.feature_names[best_feature],
                     "threshold": best_threshold, "score": float(best_score)})
        node["left"] = self._build(X[best_mask], y[best_mask], depth + 1)
        node["right"] = self._build(X[~best_mask], y[~best_mask], depth + 1)
        return node

    def _predict_one(self, row, node):
        while not node.get("leaf", True):
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.root) for row in X])

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 7))
        ax.axis("off")
        self._draw_node(ax, self.root, 0.5, 1.0, 0.25, 0)
        ax.set_title(f"{self.criterion.replace('_', ' ').title()} decision tree")
        return ax

    def _draw_node(self, ax, node, x, y, dx, depth):
        if node.get("leaf", True):
            text = f"Predict: {node['prediction']}\\nSamples: {node['samples']}"
        else:
            text = f"{node['feature_name']} ≤ {node['threshold']:.3g}\\nScore: {node['score']:.3f}"
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black"),
                transform=ax.transAxes)
        if not node.get("leaf", True):
            child_y = y - 0.14
            for child, child_x, label in [(node["left"], x-dx, "Yes"), (node["right"], x+dx, "No")]:
                ax.plot([x, child_x], [y-0.025, child_y+0.025], transform=ax.transAxes, color="black", linewidth=0.8)
                ax.text((x+child_x)/2, (y+child_y)/2, label, fontsize=7, transform=ax.transAxes)
                self._draw_node(ax, child, child_x, child_y, dx/2, depth+1)


class InformationGainTreeClassifier(_GainTree):
    def __init__(self, max_depth=8, min_samples_leaf=1):
        super().__init__("gain", max_depth, min_samples_leaf)


class GainRatioTreeClassifier(_GainTree):
    def __init__(self, max_depth=8, min_samples_leaf=1):
        super().__init__("gain_ratio", max_depth, min_samples_leaf)
