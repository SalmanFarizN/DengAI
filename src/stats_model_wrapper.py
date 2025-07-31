from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.formula.api as smf

class StatsModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, family):
        self.family = family
        self.model_ = None
        self.result_ = None

    def fit(self, X, y=None):
        # X is a pandas DataFrame, y is optional (can be in X as 'total_cases')
        data = X.copy()

        if y is not None and 'total_cases' not in data.columns:
            data['total_cases'] = y

        model_formula = (
            "total_cases ~ 1 + "
            "reanalysis_specific_humidity_g_per_kg_shift_3 + "
            "reanalysis_dew_point_temp_k_shift_3 + "
            "station_avg_temp_c_shift_3 + "
            "weekofyear_col_cosine + "
            "weekofyear_col_sine + "
            "reanalysis_avg_temp_k_shift_3"
        )

        self.model_ = smf.glm(
            formula=model_formula,
            data=data,
            family=self.family
        )
        self.result_ = self.model_.fit()
        return self

    def predict(self, X):
        return self.result_.predict(X)