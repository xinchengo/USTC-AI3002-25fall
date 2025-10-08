import numpy as np
import pandas as pd
from basemodel import LinearModel
from basetrainer import Trainer

def load_and_preprocess_data(data_file: str = "data/train.csv"):
    dataset = pd.read_csv(data_file)
    """
    Divide the dataset into features and target

    You can do all possible modifications to features, but DO NOT change the targets

    return:
        features (np.ndarray): Input features, shape [num_samples, in_features]
        targets (np.ndarray): Target values, shape [num_samples]
    """
    print(dataset.columns)
    # idxs = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB',
    #         'KWI', 'VWM', 'VWN']
    # idxs = []
    # dataset[idxs] = np.log2(dataset[idxs])
    
    features = dataset.iloc[:, :-1].to_numpy()
    targets = dataset.iloc[:, -1].to_numpy().reshape(-1, 1)
    print(f"Data size: {features.shape[0]}. Features num: {features.shape[1]}")
    return features, targets

class LinearRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Linear regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1).
        """
        self.weight = np.random.randn(in_features, out_features) * 1e-2
        self.bias = np.zeros((1, out_features))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        return features @ self.weight + self.bias

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for MSE loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): Predicted values, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        m = features.shape[0] # Batch size 
        error = predictions - targets
        dw = features.T @ error / float(m)
        db = np.sum(error, axis=0, keepdims=True) / float(m)
        return dw, db

    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute MSE loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True values, shape [batch_size, out_features].
            predictions (np.ndarray): True values, shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.
        """
        m = features.shape[0] # Batch size 
        loss = (1/(2*m)) * np.sum((predictions - targets) ** 2)
        dw, db = self.gradient(features, targets, predictions)
        np.clip(dw, -1, 1, out=dw)
        np.clip(db, -1, 1, out=db)
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        # print(f"Weight norm: {np.linalg.norm(self.weight)}, Bias norm: {np.linalg.norm(self.bias)}")
        return loss

class LinearRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.1, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=100, eval_metric="mae"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)

    def compute_loss(self, batch_pred, batch_grd):
        """
        Compute loss based on model type with detailed checks for linear regression.

        Args:
            batch_pred: Predicted values, shape [batch_size, out_features].
            batch_grd: True values/labels, shape [batch_size, out_features].

        Returns:
            float: Mean loss for the batch.
        """
        return np.mean((batch_pred - batch_grd) ** 2)

def linear_regression_analytic(X, y):
    """
    Calculate the analytical linear regression results.

    Args:
        X (np.ndarray): Input features, shape [num_samples, in_features]
        y (np.ndarray): True values, shape [num_samples, out_features]

    Return:
        weight (np.ndarray): Model weight
        bias (np.ndarray | float): Model bias
    """
    # Add a column of ones to X for the bias term
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # shape [n, d+1]
    coef = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y  # shape [d+1, 1]
    bias = coef[0].reshape(1, 1)
    weight = coef[1:]
    return weight, bias

class LogisticRegressionModel(LinearModel):
    def __init__(self, in_features: int, out_features: int):
        """
        Logistic regression model, inherits from LinearModel.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features (usually 1 for binary classification).
        """
        self.weight = np.random.randn(in_features, out_features) * 1e-2
        self.bias = np.zeros((1, out_features))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Predict the output given input.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
        """
        return self._sigmoid(features @ self.weight + self.bias)

    def gradient(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray) -> tuple:
        """
        Compute gradients for binary cross-entropy loss.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            predictions (np.ndarray): Predicted probabilities, shape [batch_size, out_features].

        Returns:
            tuple: (dw, db), gradients for weights and bias.
        """
        m = float(features.shape[0])  # Batch size
        error = predictions - targets
        dw = (1/m) * features.T @ error
        db = (1/m) * np.sum(error, axis=0, keepdims=True)
        return dw, db
    
    def backpropagation(self, features: np.ndarray, targets: np.ndarray, predictions: np.ndarray, learning_rate: float = 0.01) -> float:
        """
        Perform backpropagation, compute binary cross-entropy loss and update parameters.

        Args:
            features (np.ndarray): Input features, shape [batch_size, in_features].
            targets (np.ndarray): True labels (0 or 1), shape [batch_size, out_features].
            learning_rate (float): Learning rate, default 0.01.

        Returns:
            float: Binary cross-entropy loss for the batch.
        """
        m = float(features.shape[0])  # Batch size
        epsilon = 1e-15  # To avoid log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        dw, db = self.gradient(features, targets, predictions)
        self.weight -= learning_rate * dw
        self.bias -= learning_rate * db
        return loss
        
class LogisticRegressionTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader=None, 
                 save_dir=None, learning_rate=0.01, eval_strategy="epoch", 
                 eval_steps=100, num_epochs=10, eval_metric="f1"):
        super().__init__(model, train_dataloader, eval_dataloader, save_dir, 
                         learning_rate, eval_strategy, eval_steps, num_epochs, eval_metric)
        
    def compute_loss(self, batch_pred, batch_grd):
        m = float(batch_grd.shape[0])  # Batch size
        epsilon = 1e-15  # To avoid log(0)
        batch_pred = np.clip(batch_pred, epsilon, 1 - epsilon)
        loss = - (1/m) * np.sum(batch_grd * np.log(batch_pred) + (1 - batch_grd) * np.log(1 - batch_pred))
        return loss
