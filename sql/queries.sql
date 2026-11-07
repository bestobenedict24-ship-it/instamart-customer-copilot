-- Capstone Project Part A: SQL Analytics & Schema Verification

-- Query 1: Total Volume & Order Summary Metrics
SELECT 
    COUNT(order_id) AS total_orders,
    COUNT(DISTINCT customer_id) AS total_unique_customers,
    ROUND(AVG(order_amount), 2) AS average_order_value,
    SUM(order_amount) AS total_revenue
FROM orders;

-- Query 2: Support Ticket Issues Breakdown by Urgency
SELECT 
    issue_category,
    urgency_tier,
    COUNT(ticket_id) AS ticket_count
FROM support_tickets
GROUP BY issue_category, urgency_tier
ORDER BY ticket_count DESC;

-- Query 3: Customer Refund Distribution & Escalations
SELECT 
    status,
    COUNT(refund_id) AS total_refund_requests,
    ROUND(SUM(amount), 2) AS total_refunded_amount
FROM refunds
GROUP BY status;

-- Query 4: Identifying At-Risk Customers (High Delays & Complaints)
SELECT 
    c.customer_id,
    c.customer_name,
    COUNT(o.order_id) AS total_orders,
    SUM(CASE WHEN o.delivery_status = 'Delayed' THEN 1 ELSE 0 END) AS delayed_orders,
    COUNT(t.ticket_id) AS support_tickets_raised
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN support_tickets t ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.customer_name
HAVING support_tickets_raised > 1 OR delayed_orders > 2
ORDER BY support_tickets_raised DESC;
