
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.base import BaseEstimator, RegressorMixin


from functions import fourier_features

class FeaturesPipeline:
    def __init__(self, scaler_type, time_periods):

        if scaler_type == 'Standard':
            scaler = StandardScaler()
        else:
            scaler = None

        self.state_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", fill_value=0)),
            ("scaler", scaler),
            ("features", PolynomialFeatures(degree=1, include_bias=False)),
        ])

        self.temperature_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", fill_value=0)),
            ("scaler", scaler),
            ("features", PolynomialFeatures(degree=1, include_bias=False)),
        ])

        self.time_pipeline = Pipeline(
            [
                ("fourier",
                 FunctionTransformer(
                     fourier_features,
                     kw_args={"periods": time_periods},
                 )),
            ]
        )

        self.full_pipeline = ColumnTransformer(
            [
                ("state", self.state_pipeline, ["values"]),
                ("control", self.temperature_pipeline, ["tmp"]),
                ("time", self.time_pipeline, ["time"]),
            ]
        )

class TrainingPipeline(BaseEstimator, RegressorMixin):
    def __init__(self, time_periods=(365, 1), alpha=0.05, regressor='LSQ'):
        self.time_periods = time_periods
        self.alpha = alpha
        self.regressor = regressor
        self._build_pipeline()

    def _build_pipeline(self):
        self.training_pipeline = Pipeline(
            [
                (
                    "features_pipeline",
                    FeaturesPipeline(
                        scaler_type=None,
                        time_periods=self.time_periods,
                    ).full_pipeline,
                ),
                (
                    "regressor",
                    Lasso(alpha=self.alpha, fit_intercept=True) if self.regressor=='Lasso' else LinearRegression(fit_intercept=True),
                ),
            ]
        )
    
    def fit(self, X, y):
        self.training_pipeline.fit(X, y)
        return self
    
    def predict(self, X):
        return self.training_pipeline.predict(X)
