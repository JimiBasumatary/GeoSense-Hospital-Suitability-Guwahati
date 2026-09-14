\# Site Suitability Analysis for Hospitals in Guwahati City



\## Project Overview



This project presents a GIS, PostGIS and Machine Learning based framework for identifying suitable locations for future hospital development in Guwahati City, Assam.



The workflow integrates geospatial data processing, spatial database operations, GIS-based suitability assessment and machine learning to evaluate candidate locations based on accessibility, population, elevation, land-use characteristics and proximity to existing hospitals and commercial areas.



\## Project Title



\*\*Site Suitability Analysis for Hospitals in Guwahati City Using GIS, PostGIS and Machine Learning\*\*



\## Objectives



\- Generate candidate locations for potential hospital development.

\- Integrate OpenStreetMap, population, elevation and land-use datasets.

\- Calculate spatial accessibility and service-gap indicators.

\- Develop a GIS-based hospital suitability framework.

\- Generate training labels from the GIS-based suitability assessment.

\- Compare Random Forest and XGBoost machine learning models.

\- Interpret the selected model using SHAP.

\- Develop a reusable site-scoring module for evaluating candidate locations.

\- Perform automated unit testing using Pytest.

\- Produce a final hospital suitability dataset and GIS layer.



\## Study Area



The study focuses on \*\*Guwahati City, Assam, India\*\*.



The study area boundary was used to generate candidate locations and restrict the spatial analysis to the project area.



\## Data Used



The project integrates the following major datasets:



\- OpenStreetMap roads

\- OpenStreetMap buildings

\- OpenStreetMap hospitals/POIs

\- OpenStreetMap land-use data

\- Population raster data

\- SRTM elevation data

\- Guwahati study-area boundary



\## Software and Technologies



\- Python

\- GeoPandas

\- Rasterio

\- NumPy

\- Pandas

\- PostgreSQL

\- PostGIS

\- Scikit-learn

\- XGBoost

\- SHAP

\- Pytest

\- QGIS

\- Git

\- GitHub



\## Coordinate Reference Systems



Geographic data were handled using:



\*\*EPSG:4326 — WGS 84\*\*



Metric spatial calculations such as distances were performed using:



\*\*EPSG:32646 — WGS 84 / UTM Zone 46N\*\*



\## Methodology



```text

Guwahati Study Area

&#x20;       ↓

Spatial Data Collection

&#x20;       ↓

OpenStreetMap Data Processing

&#x20;       ↓

Population and DEM Preparation

&#x20;       ↓

Candidate Location Generation

&#x20;       ↓

PostGIS Spatial Database

&#x20;       ↓

Spatial Feature Extraction

&#x20;       ↓

Road Accessibility

&#x20;       ↓

Distance to Existing Hospitals

&#x20;       ↓

Population Value

&#x20;       ↓

Elevation

&#x20;       ↓

Commercial Accessibility

&#x20;       ↓

Land-Use Classification

&#x20;       ↓

GIS-Based Suitability Assessment

&#x20;       ↓

Training Label Creation

&#x20;       ↓

Random Forest + XGBoost

&#x20;       ↓

Model Comparison

&#x20;       ↓

XGBoost Selection

&#x20;       ↓

SHAP Interpretation

&#x20;       ↓

Reusable Site Scorer

&#x20;       ↓

Pytest Unit Testing

&#x20;       ↓

Final Hospital Suitability Layer

