import matplotlib.pyplot as plt
import numpy as np
from glum import GeneralizedLinearRegressor
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=160, centers=[(-2, -1), (2, 1)], cluster_std=0.9, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

model = GeneralizedLinearRegressor(family="binomial", alpha=0.01, l1_ratio=0, scale_predictors=True)
model.fit(X_train, y_train)

acc = np.mean((model.predict(X_test) > 0.5) == y_test)
print(f"test accuracy: {acc:.3f}")

p = model.predict(np.array([[1.5, 1.0]]))[0]
print(f"P(group 1) at (1.5, 1.0): {p:.3f}")

# predict over a grid so we can draw the 0.5 contour
g1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
g2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
xx, yy = np.meshgrid(g1, g2)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=["#dfedfc", "#fff0df"])
plt.contour(xx, yy, Z, levels=[0.5], colors="k")
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", edgecolors="w", label="train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", marker="x", label="test")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("decision_boundary.png", dpi=150)
plt.show()
