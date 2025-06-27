# Titanic Survival Prediction with PyTorch and Scikit-Learn

This project explores machine learning techniques to predict passenger survival on the Titanic, using the famous [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic). It includes:

* A **fully-connected neural network (FNN)** trained with PyTorch and evaluated with early stopping.
* A **K-Nearest Neighbors (KNN)** model optimized with `GridSearchCV` for hyperparameter tuning.
* Comparison across multiple models including Logistic Regression, Decision Trees, and MLP using scikit-learn pipelines.

## Repository Structure

| File                             | Description                                                                      |
| -------------------------------- | -------------------------------------------------------------------------------- |
| `titanic-fnn.ipynb`              | Deep learning approach using a PyTorch-based feedforward neural network.         |
| `titanic-knn-w-gridsearch.ipynb` | Classical ML approach using KNN with hyperparameter tuning and cross-validation. |


## Notebooks Overview

### `titanic-fnn.ipynb`

* Implements a PyTorch FNN with:

  * Preprocessing pipeline (imputation, scaling, encoding).
  * Early stopping based on validation loss.
  * Custom training and evaluation loops.
  * Final submission CSV for Kaggle.
* Architecture:

  * Input → Linear(14, 64) → ReLU → Linear(64, 1)
  * Binary Cross Entropy loss with logits
* Early stopping and best model checkpointing included.

Outputs:

* Accuracy and loss plots
* Test set performance
* Final submission predictions

### `titanic-knn-w-gridsearch.ipynb`

* Preprocesses data using `SimpleImputer`, `StandardScaler`, and one-hot encoding.
* Compares models using KFold cross-validation:

  * Logistic Regression
  * KNN
  * Decision Tree
  * SGDClassifier
  * MLPClassifier
* Uses `GridSearchCV` to find the best KNN configuration:

  * Optimizes `n_neighbors`, `weights`, and `metric`.

Outputs:

* Boxplots comparing accuracy and F1 score
* Final model predictions and submission file


## Technologies Used

* **Languages**: Python
* **Libraries**:

  * `PyTorch`, `torchmetrics`
  * `scikit-learn`, `pandas`, `matplotlib`, `numpy`
* **Tools**: Jupyter Notebook, Kaggle Notebooks

## How to Use

1. Clone the repo:

   ```bash
   git clone https://github.com/your-username/titanic-ml-pytorch.git
   cd titanic-ml-pytorch
   ```

2. Open in Jupyter or upload to [Kaggle](https://www.kaggle.com/code).

3. Run the notebooks in order. Ensure the Titanic dataset is available in `/kaggle/input/titanic/`.

4. Modify model architectures or pipelines to experiment!

## Results

| Model               | Best Accuracy (Val) | Notes                                 |
| ------------------- | ------------------- | ------------------------------------- |
| FNN (PyTorch)       | \~81%               | With early stopping, dropout optional |
| KNN (GridSearch)    | \~78%               | Optimized with cross-validation       |
| Logistic Regression | \~79%               | Strong baseline                       |
| MLPClassifier       | \~80%               | Comparable to FNN                     |

## Outputs

* `submission.csv`: Generated prediction file for Kaggle submission
* `fnn_ep500_earlystop_lr1e-3.pth`: Trained PyTorch model (optional for reuse)

## Author

**Farhaan Siddiqui**

## License

This project is licensed under the MIT License.
