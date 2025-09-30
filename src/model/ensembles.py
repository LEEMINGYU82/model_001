import numpy as np

class MeanEnsemble:
    def __init__(self, scalers, models):
        self.scalers = scalers
        self.models = models
        feature_list = []
        for s in scalers:
            if hasattr(s, "feature_names_in_"):
                feature_list = list(set(feature_list) | set(s.feature_names_in_))
        self.feature_names_in_ = feature_list

    def predict(self, X):
        preds = []
        for scaler, model in zip(self.scalers, self.models):
            features = getattr(scaler, "feature_names_in_", None)
            if features is None:
                continue
            if not all(feature in X.columns for feature in features):
                continue
            X_scaled = scaler.transform(X[features])
            preds.append(model.predict(X_scaled))
        return np.mean(preds, axis=0) if preds else np.zeros(len(X))
