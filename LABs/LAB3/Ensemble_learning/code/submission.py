import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from models import get_base_models
from util import bootstrap_sample
from scipy.special import expit   # sigmoid
from sklearn.linear_model import LogisticRegression


# ======================================================
# 1. Voting (Soft Voting)
# ======================================================
class VotingModel(BaseEstimator, ClassifierMixin):
    """
    实现 soft voting：
        1. 对每个基模型执行 predict_proba
        2. 对概率取平均
        3. argmax 得到最终预测
    """

    def __init__(self):
        self.models = get_base_models()
        self.model_list = list(self.models.values())

    def fit(self, X, y):
        for m in self.model_list:
            m.fit(X, y)
        return self

    def predict(self, X):
        # TODO: 实现 soft voting
        probas = [model.predict_proba(X) for model in self.model_list]
        avg_proba = np.mean(probas, axis=0)
        return np.argmax(avg_proba, axis=1)

# ======================================================
# 2. Bagging (Bootstrap + 多模型投票)
# ======================================================
class BaggingModel(BaseEstimator, ClassifierMixin):
    """
    任务：
        1. 完成 bootstrap 采样
        2. 训练多个弱分类器（DecisionTree depth=3）
        3. 对每个模型 predict 并进行多数投票
    """

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        # TODO: 实现 bootstrap 采样并训练多个弱分类器
        for i in range(self.n_estimators):
            X_sample, y_sample = bootstrap_sample(X, y)
            model = DecisionTreeClassifier(max_depth=3)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        # TODO: 实现多数投票
        from scipy import stats
        results, _ = stats.mode([h.predict(X) for h in self.models])
        return results



# ======================================================
# 3. AdaBoost (SAMME)
# ======================================================
class AdaBoostModel(BaseEstimator, ClassifierMixin):
    """
    任务：
        实现 AdaBoost 二分类（SAMME）算法主干：
            - 初始化样本权重 w
            - 训练弱分类器 stump (max_depth=1)
            - 计算加权错误率 err
            - 计算 alpha = 0.5 * log((1-err)/err)
            - 更新样本权重 w *= exp(-alpha * y_signed * pred_signed)
            - 最终使用 sign(sum alpha*h(x)) 预测
    """

    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []
        self.train_errors = [] # 记录每轮训练误差

    def fit(self, X, y):
        n = len(X)

        # 初始化样本权重
        w = np.ones(n) / n

        # 标签映射到 {-1, +1}
        y_signed = y * 2 - 1

        for t in range(self.n_estimators):

            # TODO: step 1 — 训练 stump
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X, y, sample_weight=w)

            # TODO: step 2 — 获取预测 pred & 映射 pred_signed
            pred = stump.predict(X)
            pred_signed = pred * 2 - 1

            # TODO: step 3 — 计算带权错误率 err
            err = np.dot(w, pred_signed != y_signed)

            # TODO: step 4 — 计算 alpha，应用学习率
            alpha = 0.5 * np.log((1 - err) / err)

            # TODO: step 5 — 存储 stump 和 alpha
            self.models.append(stump)
            self.alphas.append(alpha)

            # TODO: step 6 — 更新样本权重
            w = w * np.exp(-alpha * y_signed * pred_signed)
            w /= np.sum(w)

            # TODO: step 7 — 记录当前训练误差（使用自带 predict）
            self.train_errors.append(
                (self.predict(X) != y).sum() / len(y)
            )

    def predict_proba(self, X):
        return expit(
            sum(alpha * (h.predict(X) * 2 - 1)
            for alpha, h in zip(self.alphas, self.models))
        )

    def predict(self, X):
        """
        最终预测使用 sign(sum(alpha_t * h_t(x)))
        """
        # TODO:
        return (self.predict_proba(X) >= 0.5).astype(int)


# =====================================================
# 以下为选做部分
# =====================================================

# ======================================================
# 4. GBDT (简化版)
# ======================================================
class GBDTModel(BaseEstimator, ClassifierMixin):
    """
    简化版 GBDT
    使用 Logistic Loss:
        - 初始模型 F0
        - 残差 r = y - sigmoid(F)
        - 拟合回归树
        - 更新 F ← F + η f_t
    """
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.train_losses = []
        self.F0 = 0.0

    def fit(self, X, y):
        # TODO:
        # 1. 初始化 F0 = log(p/(1-p))
        # 2. for t in 1..T:
        #       p = sigmoid(F)
        #       r = y - p
        #       拟合回归树 tree.fit(X, r)
        #       F += learning_rate * tree.predict(X)
        #       记录 loss
        p0 = y.mean()
        self.F0 = np.log(p0 / (1 - p0))

        F = self.F0 * np.ones_like(y)
        for t in range(self.n_estimators):
            p = expit(F)
            r = y - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, r)
            self.trees.append(tree)

            F += self.learning_rate * tree.predict(X)
            
            p_pred = expit(F)
            loss = -(y * np.log(p_pred + 1e-15) + (1-y) * np.log(1-p_pred+1e-15)).mean()
            self.train_losses.append(loss)

    def predict_proba(self, X):
        F = (self.F0 + self.learning_rate
                * sum(t.predict(X) for t in self.trees))
        return expit(F)
            

    def predict(self, X):
        # TODO:
        # 使用 F0 + Σ η f_t(x)
        # 输出概率 sigmoid(F)
        # 返回分类结果
        return (self.predict_proba(X) >= 0.5).astype(int)


# ======================================================
# 5. Stacking (两层模型，或者你也可以实现多层)
# ======================================================
class StackingModel(BaseEstimator, ClassifierMixin):
    """
    Stacking
    """
    def __init__(self):
        from models import get_base_models
        self.base_models = get_base_models()
        self.meta_model = LogisticRegression(max_iter=200)

    def fit(self, X, y):
        # TODO:
        # 对每个 base model 训练并获得 predict_proba(X)
        # 拼接成 Z = [p1, p2, ...]
        # 用 meta_model.fit(Z, y)
        for model in self.base_models.values():
            model.fit(X, y)
        Z = np.hstack([m.predict_proba(X)[:, [1]] for m in self.base_models.values()])
        self.meta_model.fit(Z, y)
        # print(f"Coefficients of the linear regression model: \n{self.meta_model.coef_}")

    def predict(self, X):
        # TODO:
        # 同样拼接 Z_test
        # meta_model.predict(Z)
        Z_test = np.hstack([m.predict_proba(X)[:, [1]] for m in self.base_models.values()])
        return self.meta_model.predict(Z_test)
    
def truncate_adaboost_model(model, n_estimators):
    """
    截断 AdaBoost 模型，只保留前 n_estimators 个弱分类器
    用于绘制决策边界
    """
    truncated_model = AdaBoostModel(
        n_estimators=n_estimators,
        learning_rate=model.learning_rate
    )
    truncated_model.models = model.models[:n_estimators]
    truncated_model.alphas = model.alphas[:n_estimators]
    truncated_model.train_errors = model.train_errors[:n_estimators]
    return truncated_model

def truncate_gbdt_model(model, n_estimators):
    """
    截断 GBDT 模型，只保留前 n_estimators 个弱分类器
    用于绘制决策边界
    """
    truncated_model = GBDTModel(
        n_estimators=n_estimators,
        learning_rate=model.learning_rate,
        max_depth=model.max_depth,
    )
    truncated_modelF0 = model.F0
    truncated_model.trees = model.trees[:n_estimators]
    truncated_model.train_losses = model.train_losses[:n_estimators]
    return truncated_model
