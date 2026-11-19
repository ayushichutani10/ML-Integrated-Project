
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_fashion_mnist():
    train_path = BASE_DIR / "archive" / "fashion-mnist_train.csv"
    test_path = BASE_DIR / "archive" / "fashion-mnist_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Fashion-MNIST CSV files were not found in the 'archive' folder. "
            "Please make sure the dataset is downloaded and present in the project."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Training dataset shape:", train_df.shape)
    print("Testing dataset shape:", test_df.shape)
    print("\nFirst five rows:")
    print(train_df.head())

    y_train = train_df["label"].to_numpy()
    X_train = train_df.drop(columns=["label"]).to_numpy(dtype=np.float32)

    X_train = X_train / 255.0

    print("\nNumber of images:", X_train.shape[0])
    print("Number of pixel features:", X_train.shape[1])
    print("Minimum pixel value:", X_train.min())
    print("Maximum pixel value:", X_train.max())

    return train_df, test_df, y_train, X_train


def save_sample_image(X_train):
    image = X_train[0].reshape(28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.title("Fashion-MNIST Image")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_image.png", dpi=150)
    plt.close()


def run_pca_analysis(X_train):
    rng = np.random.default_rng(42)
    sample_size = min(10000, len(X_train))
    indices = rng.choice(len(X_train), size=sample_size, replace=False)
    X = X_train[indices]

    print("\n--- PCA Analysis ---")
    print("Images selected:", X.shape[0])
    print("Original features:", X.shape[1])

    pca = PCA(n_components=784, svd_solver="randomized", random_state=42)
    X_pca = pca.fit_transform(X)

    print("PCA completed successfully.")
    print("Principal components:", X_pca.shape[1])

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print("\nFirst 10 explained variance ratios:")
    print(np.round(explained_variance[:10], 4))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Scree Plot of PCA Components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scree_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, label="Cumulative variance")
    plt.axhline(0.90, color="red", linestyle="--", label="90% variance")
    plt.axhline(0.95, color="green", linestyle="--", label="95% variance")
    plt.axhline(0.99, color="orange", linestyle="--", label="99% variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance vs Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_variance.png", dpi=150)
    plt.close()

    print("\n--- Component Selection ---")
    for threshold in [0.90, 0.95, 0.99]:
        k = np.searchsorted(cumulative_variance, threshold) + 1
        print(f"Components required for {threshold * 100:.0f}% variance: {k}")

    dimensions = [10, 25, 50, 100, 200, 784]
    results = []

    for k in dimensions:
        if k > X_pca.shape[1]:
            continue

        X_reduced = X_pca[:, :k]
        X_reconstructed = X_reduced @ pca.components_[:k] + pca.mean_

        variance_retained = cumulative_variance[k - 1]
        mse = mean_squared_error(X, X_reconstructed)
        reduction = (1 - k / X.shape[1]) * 100

        results.append(
            {
                "Components": k,
                "Variance Retained (%)": variance_retained * 100,
                "Feature Reduction (%)": reduction,
                "Reconstruction MSE": mse,
            }
        )

    results_df = pd.DataFrame(results)
    print("\n--- PCA Comparison Results ---")
    print(results_df.round(4).to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / "pca_comparison_results.csv", index=False)

    display_dimensions = [10, 50, 100, 200]
    original_images = X[:5]

    fig, axes = plt.subplots(len(display_dimensions) + 1, 5, figsize=(12, 10))

    for j in range(5):
        axes[0, j].imshow(original_images[j].reshape(28, 28), cmap="gray")
        axes[0, j].set_title("Original")
        axes[0, j].axis("off")

    for row, k in enumerate(display_dimensions, start=1):
        X_reduced = X_pca[:5, :k]
        reconstructed = X_reduced @ pca.components_[:k] + pca.mean_
        reconstructed = np.clip(reconstructed, 0, 1)

        for j in range(5):
            axes[row, j].imshow(reconstructed[j].reshape(28, 28), cmap="gray")
            axes[row, j].set_title(f"{k} components")
            axes[row, j].axis("off")

    plt.suptitle("Original vs PCA-Reconstructed Images")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "image_reconstruction_comparison.png", dpi=150)
    plt.close(fig)

    print("\nAll PCA analysis completed.")
    print("Check the outputs folder for saved results.")

    return X, pca, cumulative_variance


def main():
    train_df, test_df, y_train, X_train = load_fashion_mnist()
    save_sample_image(X_train)
    run_pca_analysis(X_train)


if __name__ == "__main__":
    main()
print("Available components:", X_pca.shape[1])

# Scree plot
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot")
plt.grid(True)
plt.savefig("outputs/scree_plot.png", dpi=150)
plt.show()

# Cumulative explained variance plot
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

for threshold in [0.90, 0.95, 0.99]:
    plt.axhline(
        threshold,
        linestyle="--",
        label=f"{threshold * 100:.0f}% variance"
    )

    k = np.searchsorted(cumulative_variance, threshold) + 1
    print(f"Components for {threshold * 100:.0f}% variance: {k}")

plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance")
plt.legend()
plt.grid(True)
plt.savefig("outputs/cumulative_variance.png", dpi=150)
plt.show()

# Compare different component counts
results = []

for k in [10, 25, 50, 100, 200, 784]:
    if k > X_pca.shape[1]:
        continue

    reconstructed = (
        X_pca[:, :k] @ pca.components_[:k]
        + pca.mean_
    )

    results.append({
        "Components": k,
        "Variance Retained (%)": cumulative_variance[k - 1] * 100,
        "Feature Reduction (%)": (1 - k / 784) * 100,
        "Reconstruction MSE": mean_squared_error(X, reconstructed)
    })

results_df = pd.DataFrame(results)
print("\nPCA Comparison Results:")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "outputs/pca_comparison_results.csv",
    index=False
)

print("\nResults saved in the outputs folder.")
