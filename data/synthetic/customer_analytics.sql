select c.customer_id, o.order_id, registration_date, customer_segment, acquisition_channel, account_status,
        order_date, order_value, order_status, quantity, unit_price, category, price_tier, base_price,
        effective_date,price,pricing_strategy, pc.end_date, promotion_name,promotion_type, pro.discount_percentage,
        pro.start_date, pro.end_date 
from `customer_analytics.customers` c
left outer join `customer_analytics.orders` o
on c.customer_id = o.customer_id
left outer join `customer_analytics.order_items` i
on i.order_id = o.order_id
left outer join `customer_analytics.products` p
on p.product_id = i.product_id
left outer join `customer_analytics.product_pricing` pc
on pc.product_id = p.product_id
left outer join `customer_analytics.promotions` pro
on pro.promotion_id = o.promotion_id