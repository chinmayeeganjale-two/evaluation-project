SELECT
    issue_category,
    COUNT(*) AS complaint_count
FROM complaints
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY issue_category
ORDER BY complaint_count DESC
LIMIT 3;