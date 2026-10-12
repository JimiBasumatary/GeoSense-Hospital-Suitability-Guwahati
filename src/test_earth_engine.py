import ee

# Initialize Google Earth Engine
ee.Initialize(project='geosense-earth-engine')

print("Earth Engine initialized successfully!")

# Test dataset
image = ee.Image('USGS/SRTMGL1_003')

# Get dataset information
info = image.getInfo()

print("Dataset ID:", info['id'])

print("Google Earth Engine connection is working successfully!")
