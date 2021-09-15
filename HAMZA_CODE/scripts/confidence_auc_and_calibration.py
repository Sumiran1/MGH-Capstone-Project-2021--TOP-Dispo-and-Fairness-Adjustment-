import numpy as np
import pandas as pd
from scipy import special as sp
from typing import List


class ConfidenceIntervalsAUC:
    def __init__(self, y_true: pd.Series, y_pred: pd.Series, epsilon: float):
        self.y_true = y_true
        self.y_pred = y_pred
        self.k_0 = sum(y_true != y_pred)  # Number of errors from the classifier
        self.N = len(y_true)  # Total number of examples, s.t. N = n + m
        self.m = len(y_true[y_true == 1])  # Number of positive examples
        self.n = len(y_true[y_true == 0])  # Number of negative examples
        self.T = 2 + 3 * ((self.m - self.n)**2 + self.N)  # Qty used for calculation of variance
        self.epsilon = epsilon

    @property
    def confidence_interval_error_rate(self) -> List[float]:
        term_1 = self.k_0 / self.N
        term_2 = 1 / (2 * np.sqrt(self.N * (1 - np.sqrt(1 - self.epsilon))))
        lower_error = term_1 - term_2
        upper_error = term_1 + term_2
        return [lower_error, upper_error]

    @property
    def confidence_interval_auc(self) -> List[float]:
        k_1 = int(self.N * self.confidence_interval_error_rate[0])
        k_2 = int(self.N * self.confidence_interval_error_rate[1])
        interval_k = [x for x in range(k_1, k_2 + 1) if x > 4]
        # Max of 2000 lower & upper bounds is enough to compare
        all_lower_bounds = [self.lower_bound_auc(x) for x in interval_k][:2000]
        all_upper_bounds = [self.upper_bound_auc(x) for x in interval_k][:2000]
        lower_bound = all_lower_bounds[np.argmin(all_lower_bounds)]
        upper_bound = all_upper_bounds[np.argmax(all_upper_bounds)]
        return [lower_bound, upper_bound]

    def lower_bound_auc(self, k: int) -> float:
        return self.expectation_auc(k) - np.sqrt(self.variance_auc(k) / self.epsilon)

    def upper_bound_auc(self, k: int) -> float:
        return self.expectation_auc(k) + np.sqrt(self.variance_auc(k) / self.epsilon)

    def expectation_auc(self, k: int) -> float:
        term_1 = k/(self.m+self.n)
        term_2 = ((self.n - self.m)**2) * (self.N + 1) / (4 * self.m * self.n)
        sum_1 = sum([
            sp.binom(self.N, x)
            for x in range(0, k)
        ])
        sum_2 = sum([
            sp.binom(self.N + 1, x)
            for x in range(0, k + 1)
        ])
        expected_auc = 1 - term_1 - term_2 * (term_1 - (sum_1/sum_2))
        print("Expectation")
        print(sum_1, sum_2, sum_1/sum_2)
        print(expected_auc)
        return expected_auc

    def variance_auc(self, k: int) -> float:
        """
        Cortes and Mohri, Confidence Intervals for the Area under the ROC Curve, Corrolary 1
        """
        q_0 = (
                (self.N + 1) * self.T * (k ** 2)
                + k * (
                        self.T * (-3 * (self.n ** 2) + 3 * self.m * self.n + 3 * self.m +1)
                        - 12 * (3 * self.m * self.n + self.N)
                        - 8
                )
                + self.T * (-3 * (self.m ** 2) + 7 * self.m + 10 * self.n + 3 * self.n * self.m + 10)
                - 4 * (3 * self.m * self.n + self.N + 1)
        )
        q_1 = (
            self.T * (k ** 3)
            + 3 * (self.m - 1) * self.T * (k ** 2)
            + k * (
                    self.T * (-3 * (self.n ** 2) + 3 * self.m * self.n - 3 * self.m + 8)
                    - 6 * (6 * self.m * self.n + self.N)
            )
            + self.T * (-3 * (self.m ** 2) + 7 * self.N + 3 * self.m * self.n)
            - 2 * (6 * self.m * self.n + self.N)
        )
        term_1 = (
                         self.T * (self.N + 1) * self.N * (self.N - 1) * (
                             (self.N - 2) * self.fct_z(4, k)
                             - (2 * self.m - self.n + 3 * k - 10) * self.fct_z(3, k)
                        )
                 ) / (72 * (self.m ** 2) * (self.n ** 2))
        term_2 = (
                         self.T * self.fct_z(2, k) * (self.N + 1) * self.N * (
                            self.m ** 2 - self.n * self.m + 3 * k * self.m - 5 * self.m
                            + 2 * k ** 2 - self.n * k - 9 * k + 12
                 )
                 ) / (48 * (self.m * self.n)**2)
        term_3 = (((self.N + 1)**2) * ((self.m - self.n)**4) * ((self.fct_z(1, k))**2)) / ((4 * self.m * self.n) ** 2)
        term_4 = ((self.N + 1) * q_1 * self.fct_z(1, k)) / (72 * (self.m * self.n) ** 2)
        term_5 = k * q_0 / ((12 * self.m * self.n) ** 2)
        variance = term_1 + term_2 - term_3 - term_4 + term_5
        print("Variance")
        print(term_1, term_2, term_3, term_4, term_5)
        print(variance)
        print("-------------------------------")
        return variance

    def fct_z(self, i: int, k: int) -> float:
        """
        Function to calculate the values of Z(i) used for computation of Variance in
        Cortes and Mohri, Confidence Intervals for the Area under the ROC Curve, Corr.1
        """
        assert i < k, "i is greater than k"
        term_1 = sum([
            sp.binom(self.N + 1 - i, x)
            for x in range(0, k - i + 1)
        ])
        term_2 = sum([
            sp.binom(self.N + 1, x)
            for x in range(0, k + 1)
        ])
        print(f"Z({i, k})")
        print(term_1)
        print(term_2)
        return term_1 / term_2
