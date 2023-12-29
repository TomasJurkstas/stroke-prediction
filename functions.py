from scipy.stats import chi2_contingency, zscore
import numpy as np


def cramers_v(confusion_matrix):
    """
        Calculate Cramer's V statistic for assessing the association between
        categorical variables based on a confusion matrix.

        Parameters:
        - confusion_matrix (array-like): Contingency table representing the
          observed frequencies of the categorical variables.

        Returns:
        - float: Cramer's V statistic.

        Example:
        >>> import numpy as np
        >>> from scipy.stats import chi2_contingency
        >>> confusion_matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
        >>> cramers_v_value = cramers_v(confusion_matrix)
        >>> print(cramers_v_value)
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def drop_rows(x, y, condition):
    """
    Drop rows from input features (x) and corresponding labels (y) based on a
    specified condition.

    This function filters rows from the input features and labels based on a
    boolean condition. It is particularly useful for removing rows that meet
    certain criteria from datasets.

    Parameters:
    - x (pandas.DataFrame): Input features.
    - y (pandas.Series or pandas.DataFrame): Corresponding labels.
    - condition (pandas.Series or array-like): Boolean condition to filter
      rows. Rows corresponding to True values are dropped.

    Returns:
    - tuple: A tuple containing the filtered input features (filtered_x) and
      corresponding labels (filtered_y).
    """
    filtered_x = x.loc[~condition, :]
    filtered_y = y.loc[filtered_x.index]
    return filtered_x, filtered_y


def remove_outliers(x, y, column_name, zscore_threshold=3):
    """
    Remove outliers from input features (x) and corresponding labels (y) based
    on the Z-score of a specified column.

    This function identifies and removes outliers from the input features and
    labels by calculating the Z-score for a specified column. Outliers are
    defined as data points whose Z-score exceeds the specified threshold.

    Parameters:
    - x (pandas.DataFrame): Input features.
    - y (pandas.Series or pandas.DataFrame): Corresponding labels.
    - column_name (str): Name of the column in x for which outliers are
      detected and removed.
    - zscore_threshold (float, optional): Z-score threshold beyond which data
      points are considered outliers. Defaults to 3.

    Returns:
    - tuple: A tuple containing the filtered input features (filtered_x),
      corresponding labels (filtered_y), and a Series containing the values
      of the identified outliers.
    """
    clean_data = x[column_name].dropna()
    z_scores = zscore(clean_data)
    outliers = z_scores[abs(z_scores) > zscore_threshold]
    outlier_indexes = outliers.index
    outliers = x.loc[outlier_indexes][column_name]
    filtered_x = x.drop(outlier_indexes)
    filtered_y = y.drop(outlier_indexes)
    return filtered_x, filtered_y, outliers