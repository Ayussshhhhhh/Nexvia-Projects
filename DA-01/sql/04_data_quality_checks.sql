USE EnterpriseBI;
GO
/* Returns exception rows; zero rows is expected for the checks shown. */
SELECT 'duplicate sales grain' CheckName, COUNT_BIG(*) ExceptionCount FROM
 (SELECT SalesOrderNumber,SalesOrderLineNumber FROM stg.InternetSales GROUP BY SalesOrderNumber,SalesOrderLineNumber HAVING COUNT_BIG(*)>1) d
UNION ALL SELECT 'invalid amount or quantity',COUNT_BIG(*) FROM stg.InternetSales WHERE SalesAmount IS NULL OR SalesAmount<0 OR OrderQuantity IS NULL OR OrderQuantity<=0
UNION ALL SELECT 'missing date key',COUNT_BIG(*) FROM stg.InternetSales s LEFT JOIN AdventureWorksDW.dbo.DimDate d ON d.DateKey=s.OrderDateKey WHERE d.DateKey IS NULL
UNION ALL SELECT 'missing product key',COUNT_BIG(*) FROM stg.InternetSales s LEFT JOIN AdventureWorksDW.dbo.DimProduct d ON d.ProductKey=s.ProductKey WHERE d.ProductKey IS NULL
UNION ALL SELECT 'missing region key',COUNT_BIG(*) FROM stg.InternetSales s LEFT JOIN AdventureWorksDW.dbo.DimSalesTerritory r ON r.SalesTerritoryKey=s.SalesTerritoryKey WHERE r.SalesTerritoryKey IS NULL;
GO

CREATE OR ALTER PROCEDURE audit.usp_RunQualityChecks AS
BEGIN
 SET NOCOUNT ON;
 DECLARE @duplicates bigint=(SELECT COUNT_BIG(*) FROM (SELECT SalesOrderNumber,SalesOrderLineNumber FROM stg.InternetSales GROUP BY SalesOrderNumber,SalesOrderLineNumber HAVING COUNT_BIG(*)>1) q),
         @invalid bigint=(SELECT COUNT_BIG(*) FROM stg.InternetSales WHERE SalesAmount IS NULL OR SalesAmount<0 OR OrderQuantity IS NULL OR OrderQuantity<=0);
 IF @duplicates>0 OR @invalid>0 THROW 51004,'Staging data quality blocking exception; inspect 04_data_quality_checks.sql.',1;
END;
GO
