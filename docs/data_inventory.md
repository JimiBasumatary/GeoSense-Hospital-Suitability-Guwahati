# GeoSense Phase 1 – Exercise 3 Data Inventory
**Study Area:**
**Study Area:** Guwahati, Assam, India
\*\*Coordinate Reference System:\*\* EPSG:4326 (WGS 84)
---
\# 1. OpenStreetMap (OSM) Data
\## Buildings
\*\*Dataset:\*\* OpenStreetMap Building Footprints
\*\*Source:\*\* OpenStreetMap
\*\*Format:\*\* GeoPackage / GeoJSON
\*\*Purpose:\*\* Identification and analysis of building locations for candidate site selection.
---
\## Roads
\*\*Dataset:\*\* OpenStreetMap Road Network
\*\*Source:\*\* OpenStreetMap
\*\*Format:\*\* GeoPackage / GeoJSON
\*\*Purpose:\*\* Calculation of accessibility and distance to nearby roads.
---
\## Additional OSM Layers
Additional spatial layers were extracted from OpenStreetMap where required for the GeoSense analysis.
These datasets support spatial feature generation and candidate site evaluation.
\## Additional OSM Layers
Additional spatial layers were extracted from OpenStreetMap where required for the GeoSense analysis.
These datasets support spatial feature generation and candidate site evaluation.
\## Additional OSM Layers
Additional spatial layers were extracted from OpenStreetMap where required for the GeoSense analysis.
These datasets support spatial feature generation and candidate site evaluation.
---
\# 2. Elevation Data
\## SRTM DEM
\*\*Dataset:\*\* Shuttle Radar Topography Mission (SRTM) Digital Elevation Model
\*\*Study Area:\*\* Guwahati
\*\*File:\*\* guwahati\_srtm\_30m.tif
\*\*Spatial Resolution:\*\* Approximately 30 metres
\*\*Purpose:\*\*
\- Elevation analysis
\- Terrain analysis
\- Spatial modelling
\- Feature generation for GeoSense
---
\# 3. Population Data
\*\*Dataset:\*\* Population Raster Data
\*\*Study Area:\*\* Guwahati
\*\*Format:\*\* GeoTIFF
\*\*Purpose:\*\*
\- Population density analysis
\- Demand estimation
\- Spatial feature generation
\- Candidate site evaluation
Both raw and processed population datasets are maintained in the project data directory.
---
\# 4. Spatial Data Organisation
The GeoSense project data are organised into separate directories for:
\- Raw data
\- Processed data
\- Elevation data
\- Population data
\- OpenStreetMap data
\- Shapefiles
\- Outputs
This structure supports reproducibility and organised geospatial processing.
---
\# 5. Database
\*\*Database:\*\* PostgreSQL
\*\*Database Name:\*\* geosense\_db
\*\*Spatial Extension:\*\* PostGIS
\*\*PostGIS Version:\*\* 3.6
\*\*PostGIS Topology:\*\* Installed
The database is used for storing, managing and analysing spatial datasets required for the GeoSense project.
---
\# Exercise 3 Status
| Component | Status |
|---|---|
| Study area data | Completed |
| OSM building data | Completed |
| OSM road data | Completed |
| Additional OSM data | Completed |
| SRTM DEM | Completed |
| Population data | Completed |
| PostgreSQL database | Completed |
| PostGIS | Completed |
| PostGIS Topology | Completed |
| Data organisation | Completed |
| Data inventory documentation | Completed |
---
\## Conclusion
The required multi-source geospatial datasets for the GeoSense project have been acquired, organised and prepared for further spatial analysis and machine learning-based candidate site selection.
