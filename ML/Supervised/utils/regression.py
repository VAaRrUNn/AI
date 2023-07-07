from typing import List, Tuple
import numpy as np


def MAE(y_test: List[int | float] | Tuple[int | float] | np.array[int | float],
        y_predicted: List[int | float] | Tuple[int | float] | np.array[int | float]) -> int | float:
    """
    Implementation of mean absolute error

    MAE = difference between predicted and true values / total points

    Advantage:
        same unit as the y
        robust to outliers, i.e., not too much affected by outliers

    Drawbacks/ Disadvantages:

        Graph(of this modulus function) is not differentiable at x = 0
            This is only applicable when we use MAE as a Loss function 

    """

    return np.sum(np.abs(y_test - y_predicted)) / len(y_test)


def MSE(y_test: List[int | float] | Tuple[int | float] | np.array[int | float],
        y_predicted: List[int | float] | Tuple[int | float] | np.array[int | float]) -> int | float:
    """
    Implementation of mean square error

    MSE = square of difference between predicted and true values / total points

    Advantage:
        Can be used as a Loss function

    Drawbacks/ Disadvantages:
        Unit of the result is y^2
        Not Robust to Outliers
            Really affected by outlier cause we are taking square 

    """
    return np.sum(np.square(y_test - y_predicted)) / len(y_test)


def RMSE(y_test: List[int | float] | Tuple[int | float] | np.array[int | float],
        y_predicted: List[int | float] | Tuple[int | float] | np.array[int | float]) -> int | float:
    """
    Implementation of root mean square error

    RMSE = root of square of difference between predicted and true values / total points

    Advantage:
        Can be used as a Loss function
        Unit of result is same as y

    Drawbacks/ Disadvantages:
        Not Robust to Outliers
        Lack of interpretability
        Emphasis on large errors
            because of squaring term it gives more weight to the large errors
        Sensitive to scale
            if the target variable has a large scale then RMSE will also be large

    """
    return np.sqrt(np.sum(np.square(y_test - y_predicted)) / len(y_test))



def R2_SCORE(y_test: List[int | float] | Tuple[int | float] | np.array[int | float],
        y_predicted: List[int | float] | Tuple[int | float] | np.array[int | float]) -> int | float:
    """
    Implementation of r2_score or coefficient of determination or goodness of fit

    R2_SCORE = 1 - SSr/ SSm
    SSr -> sum of square error in the regression line
    SSm -> sum of square error in the mean line

    Bigger the better

    Advantages:
        We can interpret this score
            For eg there is a cgpa vs Lpa dataset
            and If R2_score is 0.8, It means 80% of variation in Lpa is explained by
            or is due to the cgpa part.

    Disadvantages:
        Adding more (unrellated)columns there is a increase or no change in r2_score:
            If we add temperature col to cgpa vs Lpa column then it might increase or remain same,
            which should not be the case as r2_score should have decreased as there is no kinda relation of temperatue.

        Above is handled by Adjusted r2_score

    """

    return 1 - ( np.sum(np.square(y_test - y_predicted)) ) / ( np.sum(np.square(y_test - np.mean(y_test))) )


def Adjusted_R2_SCORE(y_test: List[int | float] | Tuple[int | float] | np.array[int | float],
        y_predicted: List[int | float] | Tuple[int | float] | np.array[int | float]) -> int | float:
    """
    Implementation of Adjusted_r2_score

    Adjusted_r2_score = 1 - ( (1 - R2) (n - 1) ) / ( n - 1 - K )
    R2 -> R2_score
    n -> no of rows
    K -> total no of independent columns (more like columns in the data)

    Bigger the better (normally)

    Advantage:
        Handle the drawbacks of the R2_score
        
    """

    return 1 - ( np.sum(np.square(y_test - y_predicted)) ) / ( np.sum(np.square(y_test - np.mean(y_test))) )