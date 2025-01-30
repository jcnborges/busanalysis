USE busanalysis_dw;

SELECT base_date, COUNT(*)
FROM fat_event
GROUP BY base_date
ORDER BY base_date