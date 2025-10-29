# Master System Prompt: Berlin Road Condition → Crash Trends Analysis

## Project Context

This is a comprehensive technical specification for executing a 3-day end-to-end analysis linking street-level infrastructure quality (from Mapillary images) to road safety outcomes (Berlin crash data) for the period 2016-2024.

**Core Objective**: Build a time series of visual infrastructure features aggregated to 1000m × 1000m tiles per year, merge with annual crash counts per tile, and model Δ crashes as a function of Δ infrastructure features.

## Technical Architecture Overview

### Data Sources
1. **Mapillary API v4**: Street-level imagery metadata for Berlin (2013-2024)
2. **Berlin Unfallatlas**: Annual crash data (shapefiles/GeoJSON)
3. **OpenStreetMap**: Berlin road network extract (PBF/GeoJSON)

### Processing Pipeline
1. **Data Collection**: API queries, spatial filtering, temporal filtering
2. **OSM Infrastructure Enrichment**: Road network processing and spatial matching
3. **Tiling**: 1000m grid creation in EPSG:25833 (ETRS89 / UTM zone 33N)
4. **Spatial Joins**: Images → tiles, crashes → tiles
5. **Feature Inference**: YOLO-based visual analysis
6. **Aggregation**: Tile-year level feature computation
7. **Modeling**: Poisson regression with change features

## Key Technical Decisions

### Coordinate System
- **Primary CRS**: EPSG:25833 (ETRS89 / UTM zone 33N) for all spatial operations
- **Rationale**: Metric units, minimal distortion for Berlin area

### Tiling Strategy
- **Cluster Grid**: 15m × 15m square tiles implemented for cluster dataset
- **Coverage**: Complete Berlin administrative boundary
- **Aggregation Level**: Spatial cluster-level implemented (year-independent)

### Visual Feature Detection
- **Models**: YOLOv8s finetuned on RDD2022 Czech + Norway subset for road damage detection
- **Dataset**: 10,990 images (Czech Republic + Norway) compressed to 1024x768 pixels
- **Classes**: crack (consolidated), other corruption, Pothole
- **Compression**: 82.6% size reduction (8.4GB → 1.5GB) matching Mapillary quality
- **Features**: pothole_count, defect_area_ratio, damaged_sign_count, litter_count, green_pixels_ratio, vehicle_count
- **Sampling**: Up to k=1000 images per tile per year from tiles with ≥N images/year
- **Training Pipeline**: RDD2022 → Czech/Norway filtering → image compression → YOLO format → YOLOv8s finetuning → inference

### OSM Infrastructure Enrichment
- **Data Source**: Berlin OSM extract from Geofabrik (.pbf format)
- **Processing**: pyrosm library for efficient PBF reading
- **Road Attributes**: highway type, surface, maxspeed, lanes, lighting, cycleway, sidewalk, etc.
- **Spatial Matching**: 25m threshold for matching crashes/images to nearest road segments
- **Match Rates**: 91.3% crashes, 50.1% images successfully matched
- **Distance Column**: `dist_to_road_m` provides distance to nearest road for all points
- **Coverage**: 1.7% of images outside current OSM bounds (Brandenburg area)

### Temporal Handling
- **Feature Lag**: Always use t-1 features to predict t crashes
- **Change Features**: Δfeat_t = feat_t - feat_{t-1}
- **Temporal Analysis**: Compare "before 2020" vs "after 2020" periods using captured_at field
- **Full Temporal Range**: All available images captured regardless of date

## API Endpoints and Data Schemas

### Mapillary API v4
```
Base URL: https://graph.mapillary.com/images
Required Fields: id, created_at, geometry, sequence
Authentication: OAuth token in Authorization header
Pagination: Use 'next' cursors
Rate Limits: Respect API limits, implement backoff
```

**Example Query**:
```python
url = f"https://graph.mapillary.com/images?access_token={token}&fields=id,captured_at,geometry,thumb_256_url&bbox=13.088,52.338,13.761,52.675&limit=2000"
```

**Full Coverage Strategy**:
- Uses 32 non-overlapping tiles to cover all of Berlin (0.008 sq degrees per tile)
- Expected retrieval: 15,000-50,000 unique images
- No temporal filtering - captures all available images
- Automatic deduplication ensures no duplicate images

### Berlin Unfallatlas
- **Format**: Annual shapefiles/GeoJSON
- **Required Columns**: year, severity, lon, lat
- **Temporal Range**: 2016-2024
- **Spatial Coverage**: Berlin administrative boundary

## Code Patterns and Conventions

### Spatial Operations (GeoPandas)
```python
# Standard spatial join pattern
result = gpd.sjoin(left_gdf, right_gdf, how="inner", predicate="within")

# CRS transformation
gdf_utm = gdf.to_crs(25833)

# Bounding box operations
bbox = boundary.to_crs(25833).total_bounds
```

### Aggregation Patterns
```python
# Tile-year aggregation
agg = df.groupby(["tile_id", "year"]).agg({
    "feature": "mean",
    "count": "sum"
}).reset_index()

# Change feature computation
df = df.sort_values(["tile_id", "year"])
df["delta_feature"] = df.groupby("tile_id")["feature"].diff()
```

### API Data Handling
```python
# Pagination handling
def fetch_all_pages(base_url, params):
    all_data = []
    next_cursor = None
    
    while True:
        if next_cursor:
            params['after'] = next_cursor
        
        response = requests.get(base_url, params=params)
        data = response.json()
        all_data.extend(data['data'])
        
        if 'next' not in data:
            break
        next_cursor = data['next']
    
    return all_data
```

## Modeling Approach

### Target Variables
- **Primary**: Annual crash counts per tile (Poisson/Negative Binomial)
- **Secondary**: Crash rates per road length
- **Change Targets**: Δ crashes = crashes_t - crashes_{t-1}

### Feature Engineering
1. **Visual Features**: Raw detections, normalized by image count
2. **Change Features**: Year-over-year differences
3. **Control Variables**: images_per_tile_year, road_density, district_fixed_effects
4. **Z-score Normalization**: Within-year standardization to reduce illumination effects

### Model Specifications
```python
# Baseline model (controls only)
baseline_formula = "crashes ~ images_per_tile_year + road_density + district"

# Full model with visual features
full_formula = "crashes ~ images_per_tile_year + road_density + district + pothole_count + defect_area_ratio + damaged_sign_count + litter_count + green_pixels_ratio"

# Change model
change_formula = "delta_crashes ~ delta_pothole_count + delta_defect_area_ratio + delta_damaged_sign_count + delta_litter_count + delta_green_pixels_ratio + controls"
```

### Evaluation Framework
- **Metrics**: Log-likelihood, AIC, RMSE
- **Cross-validation**: Spatial CV (80% tiles train, 20% test)
- **Ablation Studies**: Baseline vs. + imagery vs. + Δ features
- **Marginal Gain**: Improvement from adding visual features

## Quality Checks and Validation

### Data Quality
1. **Spatial Validation**: Verify images fall within Berlin boundary
2. **Temporal Validation**: Check date ranges and temporal consistency
3. **Coverage Analysis**: Assess spatial and temporal coverage gaps
4. **Sample Validation**: Verify image sampling strategy

### Model Validation
1. **Sanity Checks**: Show class detections on random images
2. **Spatial Autocorrelation**: Test for spatial clustering in residuals
3. **Temporal Consistency**: Verify no future information leakage
4. **Robustness**: Coverage-weighted regression sensitivity

### Visualization Requirements
1. **Raw Data Analysis**: Temporal and spatial distribution of Mapillary images
   - Yearly trends (2013-2025)
   - Quarterly patterns (2013-2025) 
   - Monthly details (2018-2021)
   - Spatial distribution map with density clustering
2. **Coverage Maps**: Spatial distribution of image counts
3. **Change Heatmaps**: Δ features vs. Δ crashes
4. **Partial Dependence**: Feature effect plots
5. **Example Tiles**: Before/after infrastructure changes

## Current Project Status

### Completed
- [x] Project setup and documentation
- [x] API token configuration
- [x] Directory structure planning
- [x] Mapillary API fetcher implementation (`src/fetch/mapillary_fetch.py`)
- [x] Raw data visualization suite (`src/viz/rawdata/mapillary_viz.py`)
- [x] Traffic accident data aggregation (`src/features/aggregate_crashes.py`)
- [x] Comprehensive crash data visualization suite (`src/viz/rawdata/crashes_viz.py`)
- [x] YOLOv8 road damage detection pipeline (`src/modeling/convert_rdd_to_yolo.py`, `filter_czech_norway.py`, `compress_cz_no_dataset.py`, `train_yolo_cz_no.py`)
- [x] Crash-to-image matching (`src/features/match_crashes_to_images.py`)
- [x] Matching analysis and visualization (`src/viz/analyze_match_results.py`)
- [x] 15m cluster dataset creation (`src/features/create_cluster_dataset.py`)

### Completed Steps
1. **Data Collection**:
   - ✅ Query Mapillary API for Berlin imagery metadata (full coverage with tiling)
   - ✅ Raw data visualization (temporal and spatial analysis)
   - ✅ Download Berlin Unfallatlas crash data (2018-2021) and comprehensive visualization
   - ✅ Download OSM Berlin road network from Geofabrik (.pbf format)

2. **OSM Infrastructure Enrichment**:
   - ✅ Process OSM road network with pyrosm (190,231 segments, 12,204 km)
   - ✅ Match crashes to roads (91.3% match rate at 25m threshold)
   - ✅ Match images to roads (50.1% match rate at 25m threshold)
   - ✅ Generate comprehensive coverage analysis and Excel report
   - ✅ Create improved spatial visualizations (no redundancy)

3. **Crash-to-Image Matching**:
   - ✅ Match crashes to images based on spatial proximity (5m, 10m, 25m thresholds)
   - ✅ Filter matches by road attributes (highway AND surface must match)
   - ✅ Generate labeled datasets with crash counts, temporal info, and shared crash tracking
   - ✅ Track crashes occurring before/after/same year as image capture
   - ✅ Analyze shared crashes (crashes matched to multiple images)
   - ✅ Output: `data/processed/mapillary_labeled_{5m,10m,25m}.csv`

4. **Matching Analysis & Visualization**:
   - ✅ Generate comprehensive summary statistics comparing all thresholds
   - ✅ Analyze shared crash patterns and distributions
   - ✅ Compute temporal distributions (time differences between images and crashes)
   - ✅ Create year-wise breakdowns of matched crashes
   - ✅ Generate 8 visualization types comparing thresholds
   - ✅ Output: CSV summaries and figures in `reports/Matches/`

5. **15m Cluster Dataset Creation**:
   - ✅ Create 15m × 15m UTM grid tiles covering Berlin
   - ✅ Assign cluster_id to each image using spatial join
   - ✅ Aggregate images per cluster (year-independent clustering)
   - ✅ Match crashes to clusters using point-in-polygon (no distance threshold, no year filtering)
   - ✅ Track crash years as informational column (crash_years)
   - ✅ Generate cluster-level analysis reports and visualizations
   - ✅ Create stratified train/val/test split (70/15/15) by match_label
   - ✅ Output: `data/processed/clusters_with_crashes.csv` (11,567 clusters, 808 with crashes)
   - ✅ Output: Cluster reports and figures in `reports/Clusters/`

6. **Data Validation**:
   - ✅ Verify spatial coverage and match rates
   - ✅ Check temporal consistency
   - ✅ Assess data quality and generate recommendations

### Cluster Dataset Approach
- **Tile Size**: 15m × 15m squares in UTM (EPSG:25833)
- **Aggregation**: Year-independent clusters (all images in cluster aggregated together)
- **Crash Matching**: Point-in-polygon spatial matching only (no distance threshold, no year filtering)
- **Year Information**: Tracked as informational columns (captured_at_years, crash_years)
- **Dataset Split**: Stratified by match_label to balance positive/negative examples
- **Results**: 11,567 clusters, 808 with crashes (7.0% match rate), 1,249 total matched crashes

### Next Steps
1. **Spatial Setup**:
   - ✅ Created 15m cluster grid (implemented)
   - Perform spatial joins (images → tiles, crashes → tiles)
   - Compute coverage statistics

2. **Feature Inference**:
   - ✅ Implement YOLOv8 road damage detection pipeline
   - ✅ Convert RDD2022 dataset to YOLO format
   - ✅ Train YOLOv8s model on road damage classes
   - Process matched images for infrastructure features using trained model

### Implementation Priorities
1. **Robustness**: Handle API rate limits and data inconsistencies
2. **Reproducibility**: Document all data sources and processing steps
3. **Scalability**: Design for efficient processing of large image datasets
4. **Validation**: Implement comprehensive quality checks

## Risk Mitigation

### Technical Risks
- **API Rate Limits**: Implement exponential backoff and request batching
- **Data Quality**: Multiple validation layers and manual spot checks
- **Memory Management**: Process data in chunks, use efficient data types
- **Model Convergence**: Multiple random seeds, regularization

### Methodological Risks
- **Selection Bias**: Coverage controls and sensitivity analysis
- **Temporal Mismatch**: Strict temporal ordering and lag features
- **Spatial Autocorrelation**: Spatial CV and cluster-robust standard errors
- **Overfitting**: Regularization and cross-validation

## Success Criteria

### Technical Deliverables
- [ ] Complete data pipeline (raw → processed)
- [ ] Feature inference on ≥10k images
- [ ] Tile-year aggregated dataset
- [ ] Trained models with evaluation metrics
- [ ] Visualization suite

### Research Deliverables
- [ ] Statistical significance of visual features
- [ ] Marginal improvement over baseline
- [ ] Interpretable feature importance
- [ ] Spatial patterns in infrastructure-crash relationships

This specification serves as the comprehensive guide for executing the Berlin road infrastructure analysis project with full reproducibility and scientific rigor.
