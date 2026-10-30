# MLP Cancer Classifier 

This project is a complete implementation from scratch of a Multi-Layer Perceptron neural network in Python, designed to diagnose breast cancer tumors based on 30 medical features from the Breast Cancer Wisconsin (Diagnostic) dataset.

The network is built purely using matrix mathematics, without relying on machine learning frameworks (such as PyTorch, TensorFlow, or Scikit-Learn for model training), demonstrating the direct application of Forward Propagation, Backpropagation, and Stochastic Gradient Descent (SGD).

## Key Features
* **Pure Mathematical Implementation:** Gradient calculations, activation functions (Sigmoid, Tanh), and weight updates written directly using NumPy.
* **Interactive GUI:** A clean and full-featured interface built with Tkinter, divided into three main workflow tabs:
  * **Data:** Visualize class distributions, view feature histograms, and configure normalization (Min-Max or Z-Score).
  * **Training:** Configure hyperparameters (epochs, learning rate, hidden layer layout, initialization method) and track real-time MSE loss curves.
  * **Evaluation:** View test metrics (Accuracy, Precision, Recall, F1-Score) and generate a color-coded Confusion Matrix on unseen test data.
* **Model Serialization:** Save the trained network state (architecture, weights, normalization parameters) into a `.json` file and reload it anytime.

## Tech Stack
* **Python 3.x**
* **NumPy** — Linear algebra and matrix operations powering the neural network.
* **Pandas** — Dataset loading, cleaning, and preprocessing.
* **Matplotlib** — Data visualizations embedded directly into the Tkinter GUI.
* **Tkinter** — Desktop graphical user interface.

## Project Structure
* `retea.py` — Contains the core `MLP` class and neural network mathematics.
* `date.py` — Handles data loading, cleaning, normalization, and dataset splitting (Train/Val/Test).
* `gui.py` — Manages the graphical interface, background training threads, and real-time plotting.
* `data.csv` — The original medical dataset.
* `requirements.txt` — List of required Python packages.

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ContourWolf/MLP-Cancer-Classifier.git](https://github.com/ContourWolf/MLP-Cancer-Classifier.git)
   cd MLP-Cancer-Classifier
