-- BigQuery Dataset and Table Creation Script
-- E-commerce Churn/CLV Data Model

-- Create dataset
CREATE SCHEMA IF NOT EXISTS `leadloom-466707.customer_analytics`
OPTIONS(
  description="E-commerce customer analytics data for churn and CLV prediction",
  location="US"
);

-- 1. CUSTOMERS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.customers` (
  customer_id INT64 NOT NULL,
  registration_date DATE NOT NULL,
  customer_segment STRING NOT NULL,
  acquisition_channel STRING NOT NULL,
  geographic_region STRING NOT NULL,
  account_status STRING NOT NULL,
  preferred_contact_method STRING NOT NULL
)
PARTITION BY registration_date
CLUSTER BY customer_segment, account_status
OPTIONS(
  description="Core customer profile data"
);

-- 2. ACCOUNT_STATUS_HISTORY table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.account_status_history` (
  status_id INT64 NOT NULL,
  customer_id INT64 NOT NULL,
  status STRING NOT NULL,
  status_date DATE NOT NULL,
  reason STRING,
  changed_by_user_id INT64
)
PARTITION BY status_date
CLUSTER BY customer_id, status
OPTIONS(
  description="Customer account status change tracking"
);

-- 3. ORDERS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.orders` (
  order_id STRING NOT NULL,
  customer_id INT64 NOT NULL,
  order_date DATE NOT NULL,
  order_value NUMERIC(10,2) NOT NULL,
  order_status STRING NOT NULL,
  payment_method STRING NOT NULL,
  shipping_method STRING NOT NULL,
  discount_applied NUMERIC(8,2) DEFAULT 0.0,
  promotion_id INT64
)
PARTITION BY order_date
CLUSTER BY customer_id, order_status
OPTIONS(
  description="Customer purchase transactions"
);

-- 4. ORDER_ITEMS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.order_items` (
  order_item_id STRING NOT NULL,
  order_id STRING NOT NULL,
  product_id INT64 NOT NULL,
  quantity INT64 NOT NULL,
  unit_price NUMERIC(8,2) NOT NULL,
  discount_rate NUMERIC(5,4) DEFAULT 0.0
)
CLUSTER BY order_id, product_id
OPTIONS(
  description="Product-level order details"
);

-- 5. PRODUCTS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.products` (
  product_id INT64 NOT NULL,
  category STRING NOT NULL,
  subcategory STRING NOT NULL,
  brand STRING NOT NULL,
  price_tier STRING NOT NULL,
  seasonality_flag BOOL NOT NULL,
  base_price NUMERIC(8,2) NOT NULL,
  created_date DATE NOT NULL
)
CLUSTER BY category, price_tier
OPTIONS(
  description="Product catalog information"
);

-- 6. PRODUCT_PRICING table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.product_pricing` (
  pricing_id INT64 NOT NULL,
  product_id INT64 NOT NULL,
  effective_date DATE NOT NULL,
  price NUMERIC(8,2) NOT NULL,
  pricing_strategy STRING NOT NULL,
  end_date DATE
)
PARTITION BY effective_date
CLUSTER BY product_id, pricing_strategy
OPTIONS(
  description="Dynamic pricing history"
);

-- 7. PROMOTIONS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.promotions` (
  promotion_id INT64 NOT NULL,
  promotion_name STRING NOT NULL,
  discount_percentage NUMERIC(5,4) NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  promotion_type STRING NOT NULL
)
PARTITION BY start_date
CLUSTER BY promotion_type
OPTIONS(
  description="Marketing promotions"
);

-- 8. RETURNS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.returns` (
  return_id STRING NOT NULL,
  order_id STRING NOT NULL,
  return_date DATE NOT NULL,
  return_reason STRING NOT NULL,
  refund_amount NUMERIC(8,2) NOT NULL
)
PARTITION BY return_date
CLUSTER BY return_reason
OPTIONS(
  description="Product returns and refunds"
);

-- 9. PRODUCT_VIEWS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.product_views` (
  view_id STRING NOT NULL,
  customer_id INT64 NOT NULL,
  product_id INT64 NOT NULL,
  view_timestamp TIMESTAMP NOT NULL,
  session_id STRING NOT NULL
)
PARTITION BY DATE(view_timestamp)
CLUSTER BY customer_id, product_id
OPTIONS(
  description="Customer product viewing behavior"
);

-- 10. CUSTOMER_SUPPORT table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.customer_support` (
  ticket_id STRING NOT NULL,
  customer_id INT64 NOT NULL,
  created_date DATE NOT NULL,
  issue_type STRING NOT NULL,
  resolution_time_hours INT64,
  satisfaction_score INT64,
  status STRING NOT NULL
)
PARTITION BY created_date
CLUSTER BY customer_id, issue_type
OPTIONS(
  description="Customer support interactions"
);

-- 11. EMAIL_CAMPAIGNS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.email_campaigns` (
  campaign_record_id STRING NOT NULL,
  campaign_id STRING NOT NULL,
  customer_id INT64 NOT NULL,
  sent_date DATE NOT NULL,
  opened_date TIMESTAMP,
  clicked_date TIMESTAMP,
  unsubscribed_date TIMESTAMP,
  campaign_type STRING NOT NULL,
  promotion_id INT64
)
PARTITION BY sent_date
CLUSTER BY customer_id, campaign_type
OPTIONS(
  description="Email marketing campaign tracking"
);

-- 12. WEBSITE_SESSIONS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.website_sessions` (
  session_id STRING NOT NULL,
  customer_id INT64 NOT NULL,
  session_start TIMESTAMP NOT NULL,
  session_duration_minutes INT64 NOT NULL,
  pages_viewed INT64 NOT NULL,
  device_type STRING NOT NULL,
  referrer_source STRING
)
PARTITION BY DATE(session_start)
CLUSTER BY customer_id, device_type
OPTIONS(
  description="Website user sessions"
);

-- 13. PAYMENT_METHODS table
CREATE OR REPLACE TABLE `leadloom-466707.customer_analytics.payment_methods` (
  payment_method_id INT64 NOT NULL,
  customer_id INT64 NOT NULL,
  payment_type STRING NOT NULL,
  is_active BOOL NOT NULL,
  last_used_date DATE,
  failure_count INT64 DEFAULT 0,
  created_date DATE NOT NULL
)
PARTITION BY created_date
CLUSTER BY customer_id, payment_type
OPTIONS(
  description="Customer payment methods"
);

-- Add primary key constraints (using ENFORCE for data quality)
-- Note: BigQuery doesn't enforce these but they help with query optimization

-- Indexes for common query patterns
-- BigQuery automatically optimizes based on clustering, but we can add some views for common patterns

-- Create a view for active customers with recent activity
CREATE OR REPLACE VIEW `leadloom-466707.customer_analytics.active_customers_summary` AS
SELECT 
  c.customer_id,
  c.customer_segment,
  c.registration_date,
  c.account_status,
  COUNT(DISTINCT o.order_id) as total_orders,
  SUM(o.order_value) as total_spent,
  MAX(o.order_date) as last_order_date,
  DATE_DIFF(CURRENT_DATE(), MAX(o.order_date), DAY) as days_since_last_order
FROM `leadloom-466707.customer_analytics.customers` c
LEFT JOIN `leadloom-466707.customer_analytics.orders` o 
  ON c.customer_id = o.customer_id
WHERE c.account_status = 'active'
GROUP BY c.customer_id, c.customer_segment, c.registration_date, c.account_status;

-- Grant access permissions (adjust as needed)
-- GRANT `roles/bigquery.dataViewer` ON SCHEMA `leadloom-466707.customer_analytics` TO 'user:ngoziihemelandu@gmail.com';
-- GRANT `roles/bigquery.dataEditor` ON SCHEMA `leadloom-466707.customer_analytics` TO 'user:ngoziihemelandu@gmail.com';