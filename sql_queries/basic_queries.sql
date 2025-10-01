-- SQL Queries: Basic to Advanced Examples
-- Problem: Demonstrate common SQL patterns for data analytics

-- ============================================================================
-- 1. BASIC SELECT QUERIES
-- ============================================================================

-- Select all columns from a table
SELECT * FROM employees;

-- Select specific columns
SELECT employee_id, first_name, last_name, salary 
FROM employees;

-- Select with WHERE clause
SELECT first_name, last_name, salary 
FROM employees 
WHERE salary > 50000;

-- Select with multiple conditions
SELECT first_name, last_name, department, salary 
FROM employees 
WHERE salary > 50000 AND department = 'IT';

-- ============================================================================
-- 2. AGGREGATE FUNCTIONS
-- ============================================================================

-- Count total number of employees
SELECT COUNT(*) AS total_employees 
FROM employees;

-- Calculate average salary
SELECT AVG(salary) AS avg_salary 
FROM employees;

-- Find minimum and maximum salaries
SELECT 
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    AVG(salary) AS avg_salary
FROM employees;

-- Sum of all salaries by department
SELECT department, SUM(salary) AS total_salary 
FROM employees 
GROUP BY department;

-- ============================================================================
-- 3. GROUP BY and HAVING
-- ============================================================================

-- Count employees by department
SELECT department, COUNT(*) AS employee_count 
FROM employees 
GROUP BY department;

-- Average salary by department
SELECT 
    department, 
    AVG(salary) AS avg_salary,
    COUNT(*) AS employee_count
FROM employees 
GROUP BY department
ORDER BY avg_salary DESC;

-- Departments with more than 5 employees
SELECT department, COUNT(*) AS employee_count 
FROM employees 
GROUP BY department 
HAVING COUNT(*) > 5;

-- ============================================================================
-- 4. JOINS
-- ============================================================================

-- Inner join employees with departments
SELECT 
    e.employee_id,
    e.first_name,
    e.last_name,
    d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- Left join to include all employees
SELECT 
    e.first_name,
    e.last_name,
    d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Multiple joins
SELECT 
    e.first_name,
    e.last_name,
    d.department_name,
    p.project_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
INNER JOIN employee_projects ep ON e.employee_id = ep.employee_id
INNER JOIN projects p ON ep.project_id = p.project_id;

-- ============================================================================
-- 5. SUBQUERIES
-- ============================================================================

-- Employees earning above average salary
SELECT first_name, last_name, salary 
FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);

-- Employees in departments with high average salaries
SELECT first_name, last_name, department 
FROM employees 
WHERE department IN (
    SELECT department 
    FROM employees 
    GROUP BY department 
    HAVING AVG(salary) > 70000
);

-- ============================================================================
-- 6. WINDOW FUNCTIONS
-- ============================================================================

-- Rank employees by salary within each department
SELECT 
    first_name,
    last_name,
    department,
    salary,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS salary_rank
FROM employees;

-- Running total of salaries
SELECT 
    employee_id,
    first_name,
    salary,
    SUM(salary) OVER (ORDER BY employee_id) AS running_total
FROM employees;

-- Moving average of salary
SELECT 
    employee_id,
    salary,
    AVG(salary) OVER (
        ORDER BY employee_id 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moving_avg
FROM employees;

-- ============================================================================
-- 7. DATE FUNCTIONS
-- ============================================================================

-- Extract year and month from hire date
SELECT 
    first_name,
    hire_date,
    YEAR(hire_date) AS hire_year,
    MONTH(hire_date) AS hire_month
FROM employees;

-- Employees hired in the last year
SELECT first_name, last_name, hire_date 
FROM employees 
WHERE hire_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR);

-- Calculate tenure in years
SELECT 
    first_name,
    last_name,
    hire_date,
    TIMESTAMPDIFF(YEAR, hire_date, CURDATE()) AS years_employed
FROM employees;

-- ============================================================================
-- 8. STRING FUNCTIONS
-- ============================================================================

-- Concatenate first and last names
SELECT 
    CONCAT(first_name, ' ', last_name) AS full_name,
    salary
FROM employees;

-- Convert to uppercase
SELECT 
    UPPER(first_name) AS first_name_upper,
    LOWER(last_name) AS last_name_lower
FROM employees;

-- Extract substring
SELECT 
    first_name,
    LEFT(first_name, 1) AS initial,
    LENGTH(first_name) AS name_length
FROM employees;

-- ============================================================================
-- 9. CASE STATEMENTS
-- ============================================================================

-- Categorize salaries
SELECT 
    first_name,
    last_name,
    salary,
    CASE 
        WHEN salary < 50000 THEN 'Low'
        WHEN salary BETWEEN 50000 AND 75000 THEN 'Medium'
        WHEN salary > 75000 THEN 'High'
    END AS salary_category
FROM employees;

-- Performance rating based on multiple conditions
SELECT 
    first_name,
    last_name,
    sales_count,
    customer_rating,
    CASE 
        WHEN sales_count > 100 AND customer_rating >= 4.5 THEN 'Excellent'
        WHEN sales_count > 50 AND customer_rating >= 4.0 THEN 'Good'
        WHEN sales_count > 20 AND customer_rating >= 3.5 THEN 'Average'
        ELSE 'Needs Improvement'
    END AS performance
FROM employees;

-- ============================================================================
-- 10. COMPLEX ANALYTICS QUERIES
-- ============================================================================

-- Top 3 highest paid employees in each department
WITH ranked_employees AS (
    SELECT 
        first_name,
        last_name,
        department,
        salary,
        DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
    FROM employees
)
SELECT first_name, last_name, department, salary 
FROM ranked_employees 
WHERE rank <= 3;

-- Year-over-year sales growth
WITH yearly_sales AS (
    SELECT 
        YEAR(order_date) AS year,
        SUM(amount) AS total_sales
    FROM orders
    GROUP BY YEAR(order_date)
)
SELECT 
    year,
    total_sales,
    LAG(total_sales) OVER (ORDER BY year) AS prev_year_sales,
    ROUND(
        (total_sales - LAG(total_sales) OVER (ORDER BY year)) / 
        LAG(total_sales) OVER (ORDER BY year) * 100, 
        2
    ) AS growth_percentage
FROM yearly_sales;

-- Cohort analysis: customer retention
SELECT 
    cohort_month,
    month_number,
    COUNT(DISTINCT customer_id) AS active_customers,
    ROUND(
        100.0 * COUNT(DISTINCT customer_id) / 
        FIRST_VALUE(COUNT(DISTINCT customer_id)) OVER (
            PARTITION BY cohort_month ORDER BY month_number
        ),
        2
    ) AS retention_rate
FROM customer_cohorts
GROUP BY cohort_month, month_number
ORDER BY cohort_month, month_number;
