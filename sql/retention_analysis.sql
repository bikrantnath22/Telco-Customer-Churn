-- SQL-Compatible Logic for Customer Retention Analysis
-- Note: Assuming the table is named `customer_churn`

-- 1. Overall Retention Rate
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND((SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS overall_retention_rate
FROM customer_churn;

-- 2. Retention by Contract Type
SELECT 
    contract,
    COUNT(*) AS customer_count,
    ROUND((SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS retention_rate
FROM customer_churn
GROUP BY contract
ORDER BY retention_rate DESC;

-- 3. Retention by Tenure Group
SELECT 
    CASE 
        WHEN tenure <= 6 THEN '0-6 months'
        WHEN tenure <= 12 THEN '7-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 36 THEN '25-36 months'
        WHEN tenure <= 48 THEN '37-48 months'
        WHEN tenure <= 60 THEN '49-60 months'
        ELSE '61-72 months' 
    END AS tenure_group,
    COUNT(*) AS customer_count,
    ROUND((SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS retention_rate
FROM customer_churn
GROUP BY 1
ORDER BY MIN(tenure);

-- 4. Retention by Payment Method
SELECT 
    payment_method,
    COUNT(*) AS customer_count,
    ROUND((SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS retention_rate
FROM customer_churn
GROUP BY payment_method
ORDER BY retention_rate DESC;
