-- 1. Database Creation
CREATE DATABASE IF NOT EXISTS telco_churn_analytics;
USE telco_churn_analytics;

-- 2. Table Creation
-- Keeping the schema denormalized (flat) is highly recommended for analytics workflows.
-- It prevents expensive JOIN operations during BI reporting.
CREATE TABLE IF NOT EXISTS customer_churn (
    customer_id VARCHAR(20) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen TINYINT,
    partner VARCHAR(3),
    dependents VARCHAR(3),
    tenure INT,
    phone_service VARCHAR(3),
    multiple_lines VARCHAR(30),
    internet_service VARCHAR(30),
    online_security VARCHAR(30),
    online_backup VARCHAR(30),
    device_protection VARCHAR(30),
    tech_support VARCHAR(30),
    streaming_tv VARCHAR(30),
    streaming_movies VARCHAR(30),
    contract VARCHAR(30),
    paperless_billing VARCHAR(3),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10, 2),
    total_charges DECIMAL(10, 2),
    churn VARCHAR(3)
);

-- 3. Indexes
-- Creating indexes on categorical columns frequently used in GROUP BY clauses and WHERE filters.
CREATE INDEX idx_churn ON customer_churn(churn);
CREATE INDEX idx_contract ON customer_churn(contract);
CREATE INDEX idx_internet_service ON customer_churn(internet_service);
CREATE INDEX idx_payment_method ON customer_churn(payment_method);
CREATE INDEX idx_tenure ON customer_churn(tenure);

-- 4. CSV Loading Process
-- WARNING: You must replace 'C:/absolute/path/to/cleaned_dataset.csv' with your actual file path.
-- Depending on your MySQL configuration, you may need to use 'LOAD DATA LOCAL INFILE'.
LOAD DATA INFILE 'cleaned_dataset.csv'
INTO TABLE customer_churn
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(customer_id, gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, @v_total_charges, churn)
SET total_charges = NULLIF(TRIM(@v_total_charges), '');


-- 5. Data Validation Queries
-- Run these after the LOAD DATA INFILE step


-- Verify Row Count (Should return 7043)
SELECT 'Total Rows' AS metric, COUNT(*) AS value FROM customer_churn;

-- Verify Duplicate Customer IDs (Should return 0 rows)
SELECT customer_id, COUNT(*) AS duplicate_count
FROM customer_churn 
GROUP BY customer_id 
HAVING COUNT(*) > 1;

-- Verify NULL values in critical numeric columns (Should be 0)
SELECT 
    SUM(CASE WHEN tenure IS NULL THEN 1 ELSE 0 END) AS null_tenure,
    SUM(CASE WHEN monthly_charges IS NULL THEN 1 ELSE 0 END) AS null_monthly,
    SUM(CASE WHEN total_charges IS NULL THEN 1 ELSE 0 END) AS null_total
FROM customer_churn;

-- Verify Invalid Numerical Values (Min should be >= 0)
SELECT 
    MIN(tenure) AS min_tenure,
    MAX(tenure) AS max_tenure,
    MIN(monthly_charges) AS min_monthly,
    MIN(total_charges) AS min_total
FROM customer_churn;

-- Verify Categorical Integrity (Churn should ONLY have 'Yes' and 'No')
SELECT churn, COUNT(*) AS count
FROM customer_churn 
GROUP BY churn;
