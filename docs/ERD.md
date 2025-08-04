# 📊 E-Commerce Customer Analytics ERD

---

## 🔍 Key Features of This ERD

### 🧭 Core Customer Journey

- Customer registration → Account status tracking  
- Browsing behavior → Product views in sessions  
- Email marketing → Purchase conversion  
- Orders → Product-level details → Returns  
- Support interactions → Satisfaction tracking  

### 💰 Pricing & Promotion Intelligence

- Dynamic pricing history for price sensitivity analysis  
- Promotion tracking linked to orders and email campaigns  
- Price elasticity modeling capabilities  

### 📈 Behavioral Analytics

- Website engagement patterns  
- Email campaign effectiveness  
- Payment method preferences and failures  
- Customer support experience tracking  

**13 Tables Total**, covering the complete **B2C e-commerce customer lifecycle** for **churn and CLV prediction**.

---

## 📐 Entity-Relationship Diagram

```mermaid
erDiagram
    CUSTOMERS {
        bigint customer_id PK
        date registration_date
        varchar customer_segment
        varchar acquisition_channel
        varchar geographic_region
        varchar account_status
        varchar preferred_contact_method
    }

    ACCOUNT_STATUS_HISTORY {
        bigint status_id PK
        bigint customer_id FK
        varchar status
        date status_date
        varchar reason
        bigint changed_by_user_id
    }

    ORDERS {
        bigint order_id PK
        bigint customer_id FK
        date order_date
        decimal order_value
        varchar order_status
        varchar payment_method
        varchar shipping_method
        decimal discount_applied
        bigint promotion_id FK
    }

    ORDER_ITEMS {
        bigint order_item_id PK
        bigint order_id FK
        bigint product_id FK
        int quantity
        decimal unit_price
        decimal discount_rate
    }

    PRODUCTS {
        bigint product_id PK
        varchar category
        varchar subcategory
        varchar brand
        varchar price_tier
        boolean seasonality_flag
        decimal base_price
        date created_date
    }

    PRODUCT_PRICING {
        bigint pricing_id PK
        bigint product_id FK
        date effective_date
        decimal price
        varchar pricing_strategy
        date end_date
    }

    PROMOTIONS {
        bigint promotion_id PK
        varchar promotion_name
        decimal discount_percentage
        date start_date
        date end_date
        varchar promotion_type
    }

    RETURNS {
        bigint return_id PK
        bigint order_id FK
        date return_date
        varchar return_reason
        decimal refund_amount
    }

    PRODUCT_VIEWS {
        bigint view_id PK
        bigint customer_id FK
        bigint product_id FK
        timestamp view_timestamp
        varchar session_id
    }

    CUSTOMER_SUPPORT {
        bigint ticket_id PK
        bigint customer_id FK
        date created_date
        varchar issue_type
        int resolution_time_hours
        int satisfaction_score
        varchar status
    }

    EMAIL_CAMPAIGNS {
        bigint campaign_record_id PK
        bigint campaign_id
        bigint customer_id FK
        date sent_date
        timestamp opened_date
        timestamp clicked_date
        timestamp unsubscribed_date
        varchar campaign_type
        bigint promotion_id FK
    }

    WEBSITE_SESSIONS {
        varchar session_id PK
        bigint customer_id FK
        timestamp session_start
        int session_duration_minutes
        int pages_viewed
        varchar device_type
        varchar referrer_source
    }

    PAYMENT_METHODS {
        bigint payment_method_id PK
        bigint customer_id FK
        varchar payment_type
        boolean is_active
        date last_used_date
        int failure_count
        date created_date
    }

    %% Primary Relationships
    CUSTOMERS ||--o{ ACCOUNT_STATUS_HISTORY : has_history
    CUSTOMERS ||--o{ ORDERS : places
    CUSTOMERS ||--o{ PRODUCT_VIEWS : views
    CUSTOMERS ||--o{ CUSTOMER_SUPPORT : contacts
    CUSTOMERS ||--o{ EMAIL_CAMPAIGNS : receives
    CUSTOMERS ||--o{ WEBSITE_SESSIONS : creates
    CUSTOMERS ||--o{ PAYMENT_METHODS : uses

    %% Order Relationships
    ORDERS ||--o{ ORDER_ITEMS : contains
    ORDERS ||--o{ RETURNS : may_have
    ORDERS }o--|| PROMOTIONS : uses_promotion

    %% Product Relationships
    PRODUCTS ||--o{ ORDER_ITEMS : sold_as
    PRODUCTS ||--o{ PRODUCT_VIEWS : viewed_as
    PRODUCTS ||--o{ PRODUCT_PRICING : has_pricing

    %% Promotion Relationships
    PROMOTIONS ||--o{ EMAIL_CAMPAIGNS : featured_in

    %% Session Relationships
    WEBSITE_SESSIONS ||--o{ PRODUCT_VIEWS : includes
