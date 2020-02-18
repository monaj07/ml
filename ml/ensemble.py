# Gradient Boosted Machines
from ml.base import Algorithm
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier


class GBRegression(Algorithm):
    def __init__(self, data=None, order=1, n_estimators=500, max_depth=4):
        super().__init__(data=data, method="gbr", order=order)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def fit_gbr(self):
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
        self.model.fit(self.x, self.y)
        # mse = mean_squared_error(y_test, clf.predict(X_test))

    def predict(self, data=None):
        super().predict(data)
        predictions = self.model.predict(self.x)
        return predictions


class GBClassification(Algorithm):
    def __init__(self, data=None, order=1, n_estimators=500, max_depth=4):
        super().__init__(data=data, order=order, method="gbc")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
    
    def fit_gbc(self):
        self.model.fit(self.x, self.y.squeeze())

    def predict(self, data=None):
        super().predict(data)
        predictions = self.model.predict(self.x)
        return predictions


class RandomForest(Algorithm):
    def __init__(self, data=None, n_estimators=500, order=1, max_depth=None):
        super().__init__(data=data, method="rf", order=1)
        self.n_estimators = n_estimators
        self.x = data[0]
        self.y = data[1]
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    def fit_rf(self):
        self.model.fit(self.x, self.y.squeeze())

    def predict(self, data=None):
        super().predict(data)
        predictions = self.model.predict(self.x)
        return predictions