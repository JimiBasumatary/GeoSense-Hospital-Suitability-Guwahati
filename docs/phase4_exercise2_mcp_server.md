

\# Phase IV — Exercise 2: Building the MCP Server with Seven Tools



\## 1. Objective



The objective of Exercise 2 is to expose GeoSense geospatial analysis capabilities through a Model Context Protocol (MCP) server using Streamable HTTP. The server registers seven tools that can be discovered and invoked by an MCP client.



\## 2. Implementation



The main implementation is located in:



\- `src/phase4\_agent/mcp\_server.py` — MCP tool definitions and HTTP application.

\- `src/phase4\_agent/tools/common.py` — shared database, Earth Engine, validation and JSON-response helpers.

\- `src/phase4\_agent/mcp\_client\_demo.py` — client connection and tool-discovery demonstration.

\- `src/phase4\_agent/call\_all\_tools.py` — HTTP integration test for all seven tools.

\- `src/phase4\_agent/test\_tools.py` — tool-registration and execution tests.

\- `postman\_collection.json` — initial Postman collection for MCP initialization and tool discovery.



The MCP endpoint is:



`http://127.0.0.1:8002/mcp/`



The server is started from the repository root using:



```bash

python -m uvicorn src.phase4\_agent.mcp\_server:app --host 127.0.0.1 --port 8002 --log-level debug

```



\## 3. Registered tools



| Tool | Purpose |

|---|---|

| `mcp\_score\_location` | Scores a location using the Phase I hospital site-suitability model. |

| `mcp\_classify\_imagery` | Returns land-cover classification and spectral-index results. |

| `mcp\_lidar\_profile` | Handles a geographic bounding box for LiDAR profiling. |

| `mcp\_streetview\_scene` | Handles a location request for Street View scene analysis. |

| `mcp\_get\_climate\_risk` | Retrieves climate values from ERA5-Land. |

| `demographics` | Retrieves a WorldPop raster value. |

| `mcp\_find\_similar\_sites` | Finds nearby candidate locations in PostGIS. |



\## 4. Validation



Shared validation helpers check geographic coordinates and bounding boxes.



The following checks were executed:



\- Valid Guwahati coordinates were accepted.

\- Latitude 100 was rejected with a `ValueError`.

\- A bounding box with minimum latitude greater than maximum latitude was rejected with a `ValueError`.



\## 5. Testing and results



\### MCP registration and execution



The command below was executed:



```bash

set PYTHONPATH=src

python src/phase4\_agent/test\_tools.py

```



\*\*Observed result:\*\* Seven expected tools were registered, no unexpected tools were found, and all seven calls returned `is\_error: False`.



\### HTTP integration



The command below was executed:



```bash

python src/phase4\_agent/call\_all\_tools.py

```



\*\*Observed result:\*\* Seven tools were discovered and all seven HTTP tool calls completed without MCP errors.



\### Client demonstration



The command below was executed:



```bash

python src/phase4\_agent/mcp\_client\_demo.py

```



\*\*Observed result:\*\* The client initialized a session, discovered seven tools, and successfully called `mcp\_find\_similar\_sites`.



\### Example location



The test location was:



\- Latitude: `26.1445`

\- Longitude: `91.7362`



The site-scoring tool returned a suitability category of `High` with confidence `0.9994` for this test location. This is a model output for the tested coordinates, not a general conclusion about all sites in Guwahati.



\## 6. Data availability and limitations



The LiDAR and Street View tools returned structured responses with `status: data\_unavailable`.



\- \*\*LiDAR:\*\* No usable LAS/LAZ dataset or configured Phase III LiDAR module was available in the current repository.

\- \*\*Street View:\*\* No usable street-level imagery or configured analysis module was available.



These tools handled missing data without producing an MCP execution error. However, actual LiDAR profiling and Street View scene analysis have not been demonstrated.



The image-classification output identifies its selected model as `foundation\_resnet50\_substitute`. It should not be described as a successful Prithvi fine-tuned inference.



The similar-sites tool ranks nearby candidates by geographic distance; it does not calculate a feature-based similarity score.



The climate and population tools return source dataset values. Units and raster semantics should be verified before interpreting those values as risk categories or exact population counts.



\## 7. Postman



The file `postman\_collection.json` contains requests for MCP initialization and tool discovery. Its JSON syntax was validated with:



```bash

python -m json.tool postman\_collection.json

```



This confirms JSON validity only. Successful execution of the collection in Postman remains to be verified. Additional session-aware requests are required before the collection can be considered a complete seven-tool Postman test suite.



\## 8. Environment observations



During execution, Rasterio emitted PROJ database-version and CRS-definition warnings. They did not prevent the recorded tool tests from passing, but the geospatial environment should be reviewed for compatibility before relying on reprojection-sensitive operations.



\## 9. Conclusion



The GeoSense Phase IV MCP server registers seven tools and exposes them through Streamable HTTP. The recorded in-process tests, HTTP integration test, and client demonstration completed without MCP tool errors.



The implementation provides a working MCP interface for the available GeoSense capabilities. LiDAR and Street View remain unavailable-data handlers until suitable datasets and analysis modules are configured. Postman execution, additional end-to-end integration checks, and final evidence capture remain outstanding.

## 10. Final Verification and Evidence

### Automated HTTP Integration Test

The HTTP integration test discovered seven MCP tools and attempted to call all seven. The test summary reported zero failed MCP calls.

Evidence file: `docs/phase4_exercise2_http_test.txt`

### Client and Input Validation

The MCP client initialized successfully, discovered all seven tools, and called `mcp_find_similar_sites` successfully. Coordinate validation accepted valid coordinates and rejected an out-of-range latitude. Bounding-box validation rejected reversed latitude bounds.

### Manual Postman Test

The MCP initialization request returned HTTP `200 OK`. The subsequent `tools/list` request returned `400 Bad Request` because the session ID was missing. Manual tool discovery therefore remains unverified in Postman; the automated HTTP integration test provides the verified tool-execution evidence.

### Remaining Limitations

LiDAR profiling and Street View scene analysis returned `data_unavailable` because the required datasets or analysis modules were not configured. These capabilities require further configuration and have not been demonstrated with actual analysis results.

### Conclusion

The seven-tool MCP server passed its recorded automated in-process and HTTP integration tests. The client demonstration and input validation also passed. Postman initialization succeeded, but the session-dependent tool-list request remains unresolved.
