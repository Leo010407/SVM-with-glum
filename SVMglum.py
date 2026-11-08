"""Fit a binomial GLM to labeled points and plot its decision boundary."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from glum import GeneralizedLinearRegressor
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--no-show", action="store_true", help="Save the plot without opening a window.")
args = parser.parse_args()

# Generate two labeled groups and keep one quarter of each group for testing.
X, y = make_blobs(
    n_samples=160, centers=[(-2, -1), (2, 1)], cluster_std=0.9, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# A binomial model with a logit link predicts the probability of group 1.
model = GeneralizedLinearRegressor(
    family="binomial", link="logit", alpha=0.01, l1_ratio=0, scale_predictors=True
)
model.fit(X_train, y_train)
threshold = 0.5
test_groups = (model.predict(X_test) >= threshold).astype(int)
correct = np.count_nonzero(test_groups == y_test)
print(f"Test accuracy: {correct}/{len(y_test)} ({correct / len(y_test):.1%})")

new_point = np.array([[1.5, 1.0]])
probability = model.predict(new_point)[0]
print(f"Point {new_point[0]}: P(group 1) = {probability:.1%}, predicted group = {int(probability >= threshold)}")

# Evaluate the same probability across the plane. Its 50% contour is a line.
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 250)
x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 250)
xx, yy = np.meshgrid(x1, x2)
grid = np.column_stack([xx.ravel(), yy.ravel()])
probabilities = model.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, probabilities, levels=[0, threshold, 1], colors=["#dfedfc", "#fff0df"])
ax.contour(xx, yy, probabilities, levels=[threshold], colors="black", linewidths=2)
for group, color in [(0, "#2563eb"), (1, "#df5a08")]:
    train = X_train[y_train == group]
    test = X_test[y_test == group]
    ax.scatter(*train.T, color=color, edgecolors="white", label=f"Group {group} (training)")
    ax.scatter(*test.T, color=color, marker="X", edgecolors="black", label=f"Group {group} (test)")
ax.set(xlabel="x1", ylabel="x2", title="Logistic regression with glum")
ax.legend()
fig.tight_layout()
output = Path(__file__).with_name("decision_boundary.png")
fig.savefig(output, dpi=160)
print(f"Plot saved to: {output}")
if not args.no_show:
    plt.show()
plt.close(fig)
