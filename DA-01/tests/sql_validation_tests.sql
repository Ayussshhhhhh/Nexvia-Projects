USE EnterpriseBI;
GO
/* Reconciliation: source/stage/curated counts and amounts should match for the full reload. */
SELECT 'stage count' Test, (SELECT COUNT_BIG(*) FROM stg.InternetSales) Actual,
 (SELECT COUNT_BIG(*) FROM mart.FactSalesInternet) Expected;
SELECT 'sales sum delta' Test,
 (SELECT SUM(SalesAmount) FROM stg.InternetSales)-(SELECT SUM(SalesAmount) FROM mart.FactSalesInternet) Delta;
/* Duplicate business grain and orphan keys: all counts should be zero. */
SELECT 'duplicate line grain' Test,COUNT_BIG(*) Exceptions FROM (SELECT SalesOrderNumber,SalesOrderLineNumber FROM mart.FactSalesInternet GROUP BY SalesOrderNumber,SalesOrderLineNumber HAVING COUNT_BIG(*)>1) x;
SELECT 'orphan date' Test,COUNT_BIG(*) Exceptions FROM mart.FactSalesInternet f LEFT JOIN mart.DimDate d ON d.DateKey=f.OrderDateKey WHERE d.DateKey IS NULL;
SELECT 'orphan product' Test,COUNT_BIG(*) Exceptions FROM mart.FactSalesInternet f LEFT JOIN mart.DimProduct d ON d.ProductKey=f.ProductKey WHERE d.ProductKey IS NULL;
SELECT 'invalid amount/quantity' Test,COUNT_BIG(*) Exceptions FROM mart.FactSalesInternet WHERE SalesAmount IS NULL OR SalesAmount<0 OR OrderQuantity IS NULL OR OrderQuantity<=0;
SELECT 'orphan region' Test,COUNT_BIG(*) Exceptions FROM mart.FactSalesInternet f LEFT JOIN mart.DimRegion r ON r.RegionKey=f.RegionKey WHERE f.RegionKey IS NOT NULL AND r.RegionKey IS NULL;
/* Date continuity across the full date dimension. */
WITH d AS (SELECT FullDate, LAG(FullDate) OVER(ORDER BY FullDate) PreviousDate FROM mart.DimDate)
SELECT 'date gap' Test,COUNT_BIG(*) Exceptions FROM d WHERE PreviousDate IS NOT NULL AND DATEDIFF(day,PreviousDate,FullDate)<>1;
/* Gross profit proxy should reconcile from staged to curated totals. */
SELECT 'gross profit delta' Test,
 (SELECT SUM(SalesAmount-COALESCE(TotalProductCost,0)) FROM stg.InternetSales)
-(SELECT SUM(SalesAmount-COALESCE(TotalProductCost,0)) FROM mart.FactSalesInternet) Delta;
GO
