USE EnterpriseBI;
GO
/* Internet sales gross margin proxy: sales minus product cost. It is not GAAP operating profit. */
CREATE OR ALTER VIEW mart.vw_MonthlySales AS
WITH Monthly AS (
 SELECT d.CalendarYear,d.MonthNumber,d.YearMonth,DATEFROMPARTS(d.CalendarYear,d.MonthNumber,1) MonthStart,
  r.RegionName,p.CategoryName,
  SUM(f.SalesAmount) Revenue,SUM(COALESCE(f.TotalProductCost,0)) ProductCost,
  SUM(f.SalesAmount-COALESCE(f.TotalProductCost,0)) GrossProfit,COUNT(DISTINCT f.SalesOrderNumber) Orders
 FROM mart.FactSalesInternet f JOIN mart.DimDate d ON d.DateKey=f.OrderDateKey
 LEFT JOIN mart.DimRegion r ON r.RegionKey=f.RegionKey JOIN mart.DimProduct p ON p.ProductKey=f.ProductKey
 GROUP BY d.CalendarYear,d.MonthNumber,d.YearMonth,r.RegionName,p.CategoryName)
SELECT m.CalendarYear,m.MonthNumber,m.YearMonth,m.RegionName,m.CategoryName,m.Revenue,m.ProductCost,m.GrossProfit,m.Orders,
 m.GrossProfit/NULLIF(m.Revenue,0) MarginPct,
 pm.Revenue PriorMonthRevenue,py.Revenue PriorYearRevenue,
 (m.Revenue-pm.Revenue)/NULLIF(pm.Revenue,0) MoMGrowthPct,
 (m.Revenue-py.Revenue)/NULLIF(py.Revenue,0) YoYGrowthPct,
 RANK() OVER(PARTITION BY m.CalendarYear,m.MonthNumber ORDER BY m.Revenue DESC) RegionalRank
FROM Monthly m
LEFT JOIN Monthly pm ON pm.RegionName=m.RegionName AND pm.CategoryName=m.CategoryName AND pm.MonthStart=DATEADD(month,-1,m.MonthStart)
LEFT JOIN Monthly py ON py.RegionName=m.RegionName AND py.CategoryName=m.CategoryName AND py.MonthStart=DATEADD(year,-1,m.MonthStart);
GO
