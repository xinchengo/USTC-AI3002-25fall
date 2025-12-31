# submission.py (Student Version)

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from models import get_linear_svm, get_rbf_svm, get_poly_svm

# ================================================================
# 1. 手写核函数
# ================================================================

def rbf_kernel(X1, X2, gamma):
    """
    RBF 核函数：
        K(x, z) = exp(-gamma ||x - z||^2)

    # TODO: implement RBF kernel
        1. 计算 X1 与 X2 的欧氏距离平方矩阵
        2. 返回 exp(-gamma * dist_sq)

    提示：
        ||x - z||^2 = x^2 + z^2 - 2 x z^T
    """
    
    # X1: (n1, d)
    # X2: (n2, d)
    # result: (n1, n2)
    dist_sq = (
        (X1 ** 2).sum(1, keepdims=True)
        + (X2 ** 2).sum(1, keepdims=True).T
        - 2 * X1 @ X2.T
    )
    return np.exp(-gamma * dist_sq) 


def poly_kernel(X1, X2, degree=3, coef0=1.0, gamma=1.0):
    """
    多项式核：
        K(x, z) = (gamma x^T z + coef0)^degree

    # TODO: implement Polynomial kernel
        1. 使用 np.dot 计算 x^T z
        2. 按公式实现 poly kernel
    """
    
    # X1: (n1, d)
    # X2: (n2, d)
    return (gamma * np.dot(X1, X2.T)) ** degree

# ================================================================
# 2. Kernel 管理器
# ================================================================

class KernelManager:
    """用于管理 kernel 函数的统一接口"""

    def __init__(self, kernel="rbf", gamma="scale", degree=3, coef0=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def compute(self, X1, X2):
        """根据 kernel 类型调用对应手写核函数"""

        gamma_value = self._compute_gamma(X1) if self.gamma == "scale" else float(self.gamma)

        if self.kernel == "rbf":
            return rbf_kernel(X1, X2, gamma=gamma_value)

        elif self.kernel == "poly":
            return poly_kernel(
                X1, X2,
                degree=self.degree,
                coef0=self.coef0,
                gamma=gamma_value
            )

        elif self.kernel == "linear":
            return np.dot(X1, X2.T)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_gamma(self, X):
        """sklearn scale 策略"""
        return 1.0 / (X.shape[1] * X.var())


# ================================================================
# 3. 基于手写 kernel 的 Kernel SVM
# ================================================================

class KernelSVM(BaseEstimator, ClassifierMixin):
    """
    使用手写 kernel 的 SVM。

    内部使用 SVC(kernel="precomputed") 来训练 precomputed kernel 矩阵。
    """
    def __init__(self, kernel="rbf", C=1.0, gamma="scale", degree=3, coef0=1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.X_train = None

    def fit(self, X, y):
        """
        # TODO:
            1. 保存 X_train
            2. 调用 kernel_manager.compute(X, X) 计算核矩阵 K_train
            3. 使用 self.model.fit(K_train, y)
        """
        # Create kernel_manager and model here so GridSearchCV's set_params works correctly
        self.kernel_manager = KernelManager(
            kernel=self.kernel, gamma=self.gamma,
            degree=self.degree, coef0=self.coef0
        )
        self.model = SVC(kernel="precomputed", C=self.C, probability=True)
        
        self.X_train = X
        K_train = self.kernel_manager.compute(X, X)
        self.model.fit(K_train, y)

        # print(f"[submission.py] Trained KernelSVM with kernel={self.kernel}")
        # print(f"[submission.py] Number of support vectors: {self.model.n_support_}")

    def predict(self, X):
        """
        # TODO:
            1. 计算 K_test = K(X_test, X_train)
            2. self.model.predict(K_test)
        """
        K_test = self.kernel_manager.compute(X, self.X_train)
        return self.model.predict(K_test)

    def predict_proba(self, X):
        """
        # TODO:
            1. 同上，计算 K_test
            2. self.model.predict_proba(K_test)
        """
        K_test = self.kernel_manager.compute(X, self.X_train)
        return self.model.predict_proba(K_test)


# ================================================================
# 5. Grid Search（补全参数表）
# ================================================================

class GridSearchSVM:
    """
    完成:
        1. 参数网格 param_grid
        2. GridSearchCV 的初始化
        3. 保存 best_model
    """
    def __init__(self, kfold=5):
        self.kfold = kfold
        self.best_model = None
        self.best_params = None

    def fit(self, X, y):
        # TODO: 完成 param_grid
        param_grid = {
            'C': np.logspace(-1, 2, 10),
            'gamma': np.logspace(-2, 0, 7),
            # 'kernel': ['rbf', 'poly', 'linear'],
            'kernel': ['rbf'],
        }
        from sklearn.model_selection import GridSearchCV
        self.best_model = GridSearchCV(
            estimator=KernelSVM(),
            param_grid=param_grid,
            scoring='f1',
            cv=self.kfold,
            return_train_score=False,
            n_jobs=-1
        )
        self.best_model.fit(X, y)
        self.best_params = self.best_model.best_params_

    def predict(self, X):
        return self.best_model.predict(X)

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)


# ================================================================
# 6. 主接口
# ================================================================

def get_student_model(method, args):
    """
    供 train.py 使用，实现以下选择逻辑：

        if method == "linear":
            return Linear SVM
        if method == "rbf":
            return RBF SVM
        if method == "poly":
            return Poly SVM
        if method == "kernel_custom":
            return KernelSVM (手写核)
        if method == "grid":
            return GridSearchSVM

    # TODO:
        补全上述逻辑
    """

    if method in {'linear', 'rbf', 'poly', 'kernel_custom'}:
        return KernelSVM(method, C=args.C, gamma=args.gamma, degree=args.degree)
    elif method == 'grid':
        return GridSearchSVM(kfold=args.kfold)
    else:
        return ValueError(f"Undefined method {method}")
