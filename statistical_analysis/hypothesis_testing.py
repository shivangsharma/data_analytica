"""
Statistical Analysis: Hypothesis Testing
Problem: Perform various hypothesis tests to make statistical inferences
"""

import numpy as np
from scipy import stats


def ttest_example():
    """
    T-Test Example: Test if two groups have different means
    Problem: Are the test scores of two groups significantly different?
    """
    # Sample data: test scores from two groups
    group1 = [85, 88, 92, 78, 90, 87, 89, 91, 86, 88]
    group2 = [75, 78, 82, 73, 80, 77, 79, 81, 76, 78]
    
    # Perform independent t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    print("T-Test Example:")
    print(f"Group 1 mean: {np.mean(group1):.2f}")
    print(f"Group 2 mean: {np.mean(group2):.2f}")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Reject null hypothesis (p < {alpha})")
        print("The two groups have significantly different means.")
    else:
        print(f"Result: Fail to reject null hypothesis (p >= {alpha})")
        print("No significant difference between the two groups.")
    
    return t_statistic, p_value


def chi_square_test():
    """
    Chi-Square Test: Test independence between categorical variables
    Problem: Is there a relationship between gender and product preference?
    """
    # Observed frequencies: rows=gender, columns=product preference
    observed = np.array([
        [30, 20, 10],  # Male: Product A, B, C
        [15, 35, 20]   # Female: Product A, B, C
    ])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    print("\n\nChi-Square Test Example:")
    print("Observed frequencies:")
    print(observed)
    print(f"\nExpected frequencies:")
    print(expected)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Reject null hypothesis (p < {alpha})")
        print("Gender and product preference are dependent.")
    else:
        print(f"Result: Fail to reject null hypothesis (p >= {alpha})")
        print("Gender and product preference are independent.")
    
    return chi2, p_value


def anova_test():
    """
    ANOVA Test: Test if means of multiple groups are equal
    Problem: Do three different teaching methods result in different test scores?
    """
    # Test scores from three different teaching methods
    method1 = [85, 88, 90, 87, 89, 91, 86]
    method2 = [78, 82, 80, 79, 81, 83, 77]
    method3 = [92, 95, 93, 94, 96, 91, 93]
    
    # Perform one-way ANOVA
    f_statistic, p_value = stats.f_oneway(method1, method2, method3)
    
    print("\n\nANOVA Test Example:")
    print(f"Method 1 mean: {np.mean(method1):.2f}")
    print(f"Method 2 mean: {np.mean(method2):.2f}")
    print(f"Method 3 mean: {np.mean(method3):.2f}")
    print(f"F-statistic: {f_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Reject null hypothesis (p < {alpha})")
        print("At least one teaching method has a significantly different mean.")
    else:
        print(f"Result: Fail to reject null hypothesis (p >= {alpha})")
        print("All teaching methods have similar means.")
    
    return f_statistic, p_value


def correlation_test():
    """
    Correlation Test: Test if two variables are correlated
    Problem: Is there a correlation between study hours and test scores?
    """
    # Sample data
    study_hours = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    test_scores = [65, 70, 75, 80, 82, 85, 88, 90, 92, 95]
    
    # Pearson correlation
    correlation, p_value = stats.pearsonr(study_hours, test_scores)
    
    print("\n\nCorrelation Test Example:")
    print(f"Pearson correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Reject null hypothesis (p < {alpha})")
        if correlation > 0:
            print("There is a significant positive correlation.")
        else:
            print("There is a significant negative correlation.")
    else:
        print(f"Result: Fail to reject null hypothesis (p >= {alpha})")
        print("No significant correlation found.")
    
    return correlation, p_value


if __name__ == "__main__":
    ttest_example()
    chi_square_test()
    anova_test()
    correlation_test()
