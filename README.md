# k-Nearest Neighbors (k-NN) on MNIST from Scratch

This repository contains a custom implementation of the k-Nearest Neighbors (k-NN) algorithm to classify the MNIST handwritten digit dataset. Rather than relying on existing machine learning libraries like `scikit-learn`, this project builds the algorithm from the ground up using PyTorch tensors for data handling and computation.

## 🚀 Features

- **Custom Data Split**: Manually divides the standard MNIST training set into training (54,000 images) and validation (6,000 images) splits.
- **Custom MaxPriorityQueue**: Implements a max-heap data structure from scratch to efficiently keep track of the $k$ nearest neighbors during the distance calculations.
- **Iterative k-NN**: A foundational implementation that calculates the Euclidean distance iteratively.
- **Optimized Broadcasting k-NN**: A vectorized implementation utilizing PyTorch's tensor broadcasting (`data - item`) to compute distances significantly faster without explicit loops.
- **Full Batch Broadcasting**: Further optimizes distance calculation to handle a full batch of test images simultaneously using multidimensional tensor operations.

## 🛠️ Technology Stack
- **Python 3**
- **PyTorch** (for tensor operations and broadcasting)
- **Torchvision** (for downloading and loading the MNIST dataset)
- **NumPy** (for random indexing and data splitting)

## 📂 File Structure
- `kNN_MNIST.ipynb`: The main Google Colab Jupyter Notebook containing the data loading, data structure implementations, and the different versions of the k-NN algorithm.

## 🧠 Core Implementations

### MaxPriorityQueue
A custom class designed to store tuples of `(distance, label)`. It automatically maintains the max-heap property, allowing the algorithm to efficiently discard the furthest neighbors when finding the $k$ closest ones.

### Distance Calculation
The notebook explores three methods of computing distances:
1. `euclidian_dist(v1, v2)`: A pure Python implementation using loops and `math.sqrt`.
2. `knn_broadcasting`: Uses PyTorch's `torch.sum(diff**2, dim=1)` and `torch.topk` to find neighbors for a single test image instantly.
3. `knn_broadcasting_all`: Extends broadcasting to evaluate multiple test images against the entire training set at once using `diff = data[:, None, :] - test[None, :, :]`.

## ⚙️ How to Run
If you want to run this notebook yourself:
1. Open Google Colab.
2. Select **File > Upload notebook** and upload `kNN_MNIST.ipynb`.
3. Run the cells sequentially. The `torchvision` library will automatically download the MNIST dataset to a local `./data` folder in the Colab environment.

## 📈 Results
The output demonstrates the model correctly identifying a handwritten '7' from the test set. The progression of functions in the notebook highlights the massive performance difference between standard Python loops and optimized PyTorch tensor broadcasting.
