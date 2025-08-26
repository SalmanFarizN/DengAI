from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.formula.api as smf


class StatsModelWrapper(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for statsmodels GLM regression.
    
    This wrapper enables the use of statsmodels Generalized Linear Models (GLM)
    within scikit-learn pipelines. It's specifically designed for count data
    regression in epidemiological contexts, such as predicting dengue fever cases
    using negative binomial or Poisson family distributions.
    
    The wrapper handles the interface differences between statsmodels and
    scikit-learn, making it possible to use statistical models that may be
    more appropriate for count data than standard machine learning regressors.
    
    Attributes
    ----------
    family : statsmodels.genmod.families.family.Family
        Statistical family for the GLM (e.g., NegativeBinomial, Poisson)
    model_ : statsmodels.genmod.generalized_linear_model.GLM
        Fitted statsmodels GLM instance
    result_ : statsmodels.genmod.generalized_linear_model.GLMResults  
        Fitted model results from statsmodels
        
    Examples
    --------
    >>> import statsmodels.api as sm
    >>> from sklearn.pipeline import Pipeline
    >>> 
    >>> # Create wrapper with negative binomial family for count data
    >>> nb_model = StatsModelWrapper(family=sm.families.NegativeBinomial())
    >>> pipeline = Pipeline([('model', nb_model)])
    >>> pipeline.fit(X_train, y_train)
    """
    def __init__(self, family):
        """Initialize the statsmodels GLM wrapper.
        
        Parameters
        ----------
        family : statsmodels.genmod.families.family.Family
            Statistical distribution family for the GLM. Common choices for
            count data include:
            - sm.families.NegativeBinomial() for overdispersed count data
            - sm.families.Poisson() for standard count data
            - sm.families.Gaussian() for continuous data
            
        Examples
        --------
        >>> import statsmodels.api as sm
        >>> # For count data with overdispersion
        >>> wrapper = StatsModelWrapper(sm.families.NegativeBinomial(alpha=1e-8))
        """
        self.family = family
        self.model_ = None
        self.result_ = None

    def fit(self, X, y=None):
        """Fit the GLM model to training data.
        
        This method prepares the data for statsmodels, constructs the model
        formula, and fits the GLM. It's designed to work with dengue fever
        prediction features and handles the specific feature set used in the
        DengAI competition.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix. Expected to contain the features specified in the
            hardcoded formula or a 'total_cases' column if y is None.
        y : array-like, optional
            Target values (dengue case counts). If None, expects 'total_cases'
            column to be present in X. Default is None.
            
        Returns
        -------
        self : StatsModelWrapper
            Returns self for method chaining compatibility.
            
        Notes
        -----
        - Uses a hardcoded formula optimized for dengue fever prediction
        - Formula includes lagged meteorological features and seasonal terms
        - The model is fitted using statsmodels GLM with the specified family
        
        Raises
        ------
        KeyError
            If required features are missing from the input DataFrame.
        """
        # Prepare data for statsmodels GLM fitting
        data = X.copy()

        # Add target variable to data if provided separately
        if y is not None and "total_cases" not in data.columns:
            data["total_cases"] = y

        # Define model formula with key dengue fever prediction features
        # Formula includes lagged meteorological variables and seasonal components
        model_formula = (
            "total_cases ~ 1 + "  # Intercept
            "reanalysis_specific_humidity_g_per_kg_shift_3 + "  # Lagged humidity
            "reanalysis_dew_point_temp_k_shift_3 + "  # Lagged dew point
            "station_avg_temp_c_shift_3 + "  # Lagged temperature
            "weekofyear_col_cosine + "  # Seasonal cosine component
            "weekofyear_col_sine + "  # Seasonal sine component
            "reanalysis_avg_temp_k_shift_3"  # Lagged reanalysis temperature
        )

        # Fit the GLM using statsmodels
        self.model_ = smf.glm(formula=model_formula, data=data, family=self.family)
        self.result_ = self.model_.fit()
        return self

    def predict(self, X):
        """Generate predictions using the fitted GLM model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction. Must contain the same features
            used during training as specified in the model formula.
            
        Returns
        -------
        np.ndarray
            Predicted values (dengue case counts) as a numpy array.
            
        Notes
        -----
        - Uses the fitted statsmodels GLM result for prediction
        - Predictions are generated using the same formula as training
        - Returns raw predictions without post-processing
        
        Examples
        --------
        >>> predictions = fitted_wrapper.predict(X_test)
        >>> # May need to apply post-processing for final submission
        >>> final_predictions = np.maximum(predictions.astype(int), 0)
        """
        return self.result_.predict(X)
