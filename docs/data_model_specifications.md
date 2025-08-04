# B2C E-commerce Churn/CLV Data Model Specifications

## Table Definitions & Field Specifications

### 1. CUSTOMERS (Core Customer Profile)
```sql
customer_id         BIGINT PRIMARY KEY  -- Unique customer identifier
registration_date   DATE                -- Account creation date
customer_segment    VARCHAR(20)         -- 'high_value', 'mid_value', 'low_value'
acquisition_channel VARCHAR(30)         -- 'organic_search', 'paid_search', 'social_media', 'email_marketing', 'direct', 'referral'
geographic_region   VARCHAR(50)         -- 'North America', 'Europe', 'Asia Pacific', etc.
account_status      VARCHAR(20)         -- 'active', 'inactive', 'suspended', 'closed', 'banned'
preferred_contact_method VARCHAR(20)    -- 'email', 'sms', 'phone', 'push_notification'
```

### 2. ACCOUNT_STATUS_HISTORY (Status Change Tracking)
```sql
status_id           BIGINT PRIMARY KEY  -- Unique status change record
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
status              VARCHAR(20)         -- Same values as account_status
status_date         DATE                -- When status changed
reason              VARCHAR(100)        -- 'customer_request', 'payment_failure', 'policy_violation', 'inactivity'
changed_by_user_id  BIGINT              -- System user who made change (nullable)
```

### 3. ORDERS (Purchase Transactions)
```sql
order_id            BIGINT PRIMARY KEY  -- Unique order identifier
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
order_date          DATE                -- Purchase date
order_value         DECIMAL(10,2)       -- Total order amount
order_status        VARCHAR(20)         -- 'pending', 'confirmed', 'processing', 'shipped', 'delivered', 'cancelled', 'refunded', 'failed'
payment_method      VARCHAR(30)         -- 'credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay'
shipping_method     VARCHAR(30)         -- 'standard', 'express', 'overnight', 'pickup'
discount_applied    DECIMAL(8,2)        -- Order-level discount amount
promotion_id        BIGINT FOREIGN KEY  -- Links to PROMOTIONS (nullable)
```

### 4. ORDER_ITEMS (Product-Level Order Details)
```sql
order_item_id       BIGINT PRIMARY KEY  -- Unique line item identifier
order_id            BIGINT FOREIGN KEY  -- Links to ORDERS
product_id          BIGINT FOREIGN KEY  -- Links to PRODUCTS
quantity            INT                 -- Number of items purchased
unit_price          DECIMAL(8,2)        -- Price per unit at time of purchase
discount_rate       DECIMAL(5,4)        -- Item-level discount percentage (0.0000-1.0000)
```

### 5. PRODUCTS (Product Catalog)
```sql
product_id          BIGINT PRIMARY KEY  -- Unique product identifier
category            VARCHAR(50)         -- 'Electronics', 'Clothing', 'Home & Garden', 'Books', 'Sports'
subcategory         VARCHAR(50)         -- 'Smartphones', 'T-Shirts', 'Kitchen Appliances'
brand               VARCHAR(50)         -- Product brand name
price_tier          VARCHAR(20)         -- 'budget', 'mid_range', 'premium', 'luxury'
seasonality_flag    BOOLEAN             -- TRUE if seasonal demand pattern
base_price          DECIMAL(8,2)        -- Current standard price
created_date        DATE                -- When product was added to catalog
```

### 6. PRODUCT_PRICING (Dynamic Pricing History)
```sql
pricing_id          BIGINT PRIMARY KEY  -- Unique pricing record
product_id          BIGINT FOREIGN KEY  -- Links to PRODUCTS
effective_date      DATE                -- When price became effective
price               DECIMAL(8,2)        -- Price amount
pricing_strategy    VARCHAR(30)         -- 'regular', 'competitive_match', 'demand_based', 'clearance', 'penetration', 'premium'
end_date            DATE                -- When price ended (nullable for current price)
```

### 7. PROMOTIONS (Marketing Promotions)
```sql
promotion_id        BIGINT PRIMARY KEY  -- Unique promotion identifier
promotion_name      VARCHAR(100)        -- 'Summer Sale 2024', 'Black Friday', 'Back to School'
discount_percentage DECIMAL(5,4)        -- Discount amount (0.0000-1.0000)
start_date          DATE                -- Promotion start date
end_date            DATE                -- Promotion end date
promotion_type      VARCHAR(30)         -- 'seasonal_sale', 'flash_sale', 'clearance', 'new_customer', 'loyalty_program'
```

### 8. RETURNS (Return/Refund Data)
```sql
return_id           BIGINT PRIMARY KEY  -- Unique return identifier
order_id            BIGINT FOREIGN KEY  -- Links to ORDERS
return_date         DATE                -- Date return was processed
return_reason       VARCHAR(50)         -- 'defective', 'wrong_size', 'not_as_described', 'changed_mind', 'damaged_shipping'
refund_amount       DECIMAL(8,2)        -- Amount refunded to customer
```

### 9. PRODUCT_VIEWS (Browsing Behavior)
```sql
view_id             BIGINT PRIMARY KEY  -- Unique view record
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
product_id          BIGINT FOREIGN KEY  -- Links to PRODUCTS
view_timestamp      TIMESTAMP           -- Exact time of product view
session_id          VARCHAR(50) FOREIGN KEY -- Links to WEBSITE_SESSIONS
```

### 10. CUSTOMER_SUPPORT (Service Interactions)
```sql
ticket_id           BIGINT PRIMARY KEY  -- Unique support ticket identifier
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
created_date        DATE                -- Support ticket creation date
issue_type          VARCHAR(50)         -- 'billing', 'shipping', 'product_defect', 'account_access', 'general_inquiry'
resolution_time_hours INT               -- Hours to resolve ticket
satisfaction_score  INT                 -- 1-5 rating (nullable if no response)
status              VARCHAR(20)         -- 'open', 'in_progress', 'resolved', 'closed', 'escalated'
```

### 11. EMAIL_CAMPAIGNS (Marketing Email Tracking)
```sql
campaign_record_id  BIGINT PRIMARY KEY  -- Unique email send record
campaign_id         BIGINT              -- Groups related campaign sends
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
sent_date           DATE                -- Date email was sent
opened_date         TIMESTAMP           -- When email was opened (nullable)
clicked_date        TIMESTAMP           -- When email link was clicked (nullable)
unsubscribed_date   TIMESTAMP           -- When customer unsubscribed (nullable)
campaign_type       VARCHAR(30)         -- 'promotional', 'retention', 'newsletter', 'abandoned_cart', 'cross_sell'
promotion_id        BIGINT FOREIGN KEY  -- Links to PROMOTIONS (nullable)
```

### 12. WEBSITE_SESSIONS (Digital Engagement)
```sql
session_id          VARCHAR(50) PRIMARY KEY -- Unique session identifier
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
session_start       TIMESTAMP           -- Session start time
session_duration_minutes INT            -- Total session length
pages_viewed        INT                 -- Number of pages in session
device_type         VARCHAR(20)         -- 'desktop', 'mobile', 'tablet'
referrer_source     VARCHAR(50)         -- 'google.com', 'facebook.com', 'email', 'direct', 'instagram.com'
```

### 13. PAYMENT_METHODS (Stored Payment Options)
```sql
payment_method_id   BIGINT PRIMARY KEY  -- Unique payment method identifier
customer_id         BIGINT FOREIGN KEY  -- Links to CUSTOMERS
payment_type        VARCHAR(30)         -- 'credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay', 'buy_now_pay_later'
is_active           BOOLEAN             -- Whether method is currently usable
last_used_date      DATE                -- Most recent usage date
failure_count       INT                 -- Number of failed payment attempts
created_date        DATE                -- When payment method was added
```

## Key Business Rules & Constraints

### Data Integrity Rules
- Every ORDER must have at least one ORDER_ITEM
- PRODUCT_VIEWS can only occur within valid WEBSITE_SESSIONS
- EMAIL_CAMPAIGNS with promotion_id must link to active PROMOTIONS
- ACCOUNT_STATUS_HISTORY must maintain chronological order per customer

### Temporal Constraints
- order_date ≤ return_date (returns happen after orders)
- promotion start_date < end_date
- pricing effective_date ≤ end_date
- session_start ≤ view_timestamp (views occur during sessions)

### Business Logic
- Customer segments calculated based on CLV and purchase behavior
- Account status changes trigger ACCOUNT_STATUS_HISTORY entries
- Pricing changes create new PRODUCT_PRICING entries
- Order discounts can combine order-level and item-level discounts

## External Data Integration Points

### Economic Indicators (FRED API)
- Consumer confidence index
- Unemployment rate
- Inflation rate
- GDP growth rate

### Seasonal/Calendar Data
- Holiday calendars
- School calendars
- Weather data
- Local events

### Competitive Intelligence
- Competitor pricing data
- Market share data
- Industry benchmarks

This data model provides the foundation for comprehensive churn prediction and customer lifetime value modeling in a B2C e-commerce environment.