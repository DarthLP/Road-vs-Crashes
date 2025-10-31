# Master System Prompt: Berlin Road Condition → Crash Trends Analysis

## Project Context

This is a comprehensive technical specification for executing a cross-sectional spatial analysis linking street-level infrastructure quality (from Mapillary images) to road safety outcomes (Berlin crash data) for the period 2018-2024.

**Core Objective**: Build a static spatial model of visual infrastructure features aggregated to 15 m × 15 m clusters, merge with crash occurrence per cluster, and model crash risk as a function of infrastructure and visual damage features using a static spatial framework with contemporaneous crash and imagery data.

## Technical Architecture Overview

### Data Sources
1. **Mapillary API v4**: Street-level imagery metadata for Berlin (2013-2025)
2. **Berlin Unfallatlas**: Crash data (shapefiles/GeoJSON) for 2018-2024
3. **OpenStreetMap**: Berlin road network extract (PBF/GeoJSON)

### Processing Pipeline
1. **Data Collection**: API queries, spatial filtering
2. **OSM Infrastructure Enrichment**: Road network processing and spatial matching
3. **Clustering**: 15 m × 15 m cluster grid creation in EPSG:25833 (ETRS89 / UTM zone 33N)
4. **Spatial Joins**: Images → clusters, crashes → clusters (point-in-polygon)
5. **Feature Inference**: YOLO-based visual analysis
6. **Aggregation**: Cluster-level feature computation
7. **Modeling**: Logistic regression with infrastructure and visual features

## Key Technical Decisions

### Coordinate System
- **Primary CRS**: EPSG:25833 (ETRS89 / UTM zone 33N) for all spatial operations
- **Rationale**: Metric units, minimal distortion for Berlin area

### Clustering Strategy
- **Cluster Grid**: 15 m × 15 m square clusters implemented
- **Coverage**: Complete Berlin administrative boundary
- **Aggregation Level**: Spatial cluster-level (year-independent, cross-sectional analysis)

### Visual Feature Detection
- **Models**: YOLOv8s finetuned on RDD2022 Czech + Norway subset for road damage detection
- **Dataset**: 10,990 images (Czech Republic + Norway) compressed to 1024x768 pixels
- **Test Dataset**: Czech + Norway images from RDD2022 for evaluation (1,633 test images)
- **Classes**: crack (consolidated), other corruption, Pothole
- **Compression**: 82.6% size reduction (8.4GB → 1.5GB) matching Mapillary quality
- **Features**: pothole_count, defect_area_ratio, damaged_sign_count, litter_count, green_pixels_ratio, vehicle_count
- **Sampling**: One representative image per cluster for CNN analysis
- **Training Pipeline**: RDD2022 → Czech/Norway filtering → image compression → YOLO format → YOLOv8s finetuning → inference
- **Test Pipeline**: RDD2022 → Czech/Norway filtering → image compression → test split added to dataset structure

### CNN Residual Augmentation Evaluation
- **Objective**: Quantify added value of CNN residual predictor beyond baseline logistic probabilities (and YOLO-enhanced).
- **Method**:
  - Correlate CNN predicted residuals with true residuals on Test.
  - Fit augmentation logistic regression on Val: Crash ~ p_baseline + yhat_CNN (and YOLO variant).
  - Evaluate on Test: ΔAUC and ΔMcFadden pseudo-R²; thresholds (0.5, Val-opt F1, Val-opt Youden) → accuracy/precision/recall/F1/ROC AUC.
- **Artifacts**:
  - Script: `src/modeling/evaluate_cnn_augmentation.py`
  - Outputs: CSV summaries under `reports/CNN/` and figures under `reports/Comparison_Baseline_vs_YOLO/`.
  - Usage: `conda activate berlin-road-crash && python src/modeling/evaluate_cnn_augmentation.py --use_saved_preds`

### OSM Infrastructure Enrichment
- **Data Source**: Berlin OSM extract from Geofabrik (.pbf format)
- **Processing**: pyrosm library for efficient PBF reading
- **Road Attributes**: highway type, surface, maxspeed, lanes, lighting, cycleway, sidewalk, etc.
- **Spatial Matching**: 10 m threshold for matching crashes/images to nearest road segments (OSM snapping)
- **Match Rates**: 91.3% crashes, 50.1% images successfully matched
- **Distance Column**: `dist_to_road_m` provides distance to nearest road for all points
- **Coverage**: 1.7% of images outside current OSM bounds (Brandenburg area)
- **Note**: Cluster assignment uses point-in-polygon (15 m clusters), separate from OSM road snapping (10 m threshold)

### Temporal Metadata Tracking
- **Cross-Sectional Model**: Analysis uses contemporaneous crash and imagery data (static spatial framework)
- **Temporal Information**: Track capture years as metadata (captured_at_years, crash_years columns)
- **Optional Analysis**: Can compare periods like "before 2020" vs "after 2020" using captured_at field for exploration

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
- Expected retrieval: 15,000-50,000 unique images (2013-2025)
- No temporal filtering - captures all available images
- Automatic deduplication ensures no duplicate images

### Berlin Unfallatlas
- **Format**: Annual shapefiles/GeoJSON
- **Required Columns**: year, severity, lon, lat
- **Temporal Range**: 2018-2024 (97,511 total crashes)
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
# Cluster-level aggregation
agg = df.groupby(["cluster_id"]).agg({
    "feature": "median",
    "count": "count"
}).reset_index()

# Spatial join pattern (point-in-polygon for crash assignment)
matched = gpd.sjoin(crashes_gdf, clusters_gdf, how="left", predicate="within")
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

### Baseline Model (Implemented)
- **Model Type**: Logistic Regression (binary classification)
- **Target**: `match_label` (1 = crash occurred in cluster, 0 = no crash)
- **Features**: Structured road infrastructure attributes only (18 features → 66 after one-hot encoding)
  - Categorical: highway, surface, lit, cycleway, sidewalk, oneway, bridge, tunnel, junction, access, smoothness
  - Boolean: is_intersection, near_traffic_signal
  - Numeric: road_segment_length_m, maxspeed, lanes, width, intersection_degree
- **Preprocessing**: One-hot encoding for categoricals, median imputation for numerics (train-only)
- **Performance**: ROC AUC ~0.75, Precision ~0.14, Recall ~0.68-0.76, F1 ~0.24
- **Residuals**: Computed as `actual_label - predicted_probability` for CNN regression target
- **Purpose**: Establish baseline explainable by structured data; residuals capture visual risk unexplained by OSM attributes

### Enhanced Model with YOLO Features (Implemented)
- **Model Type**: Logistic Regression (binary classification) + YOLO visual features
- **Target**: `match_label` (same as baseline)
- **Features**: All baseline features + `street_quality_roi_winz` (primary YOLO feature)
- **YOLO Feature Extraction**:
  - ROI Filtering: Square ROI covering the bottom 60% of the image height, centered horizontally to focus on road area
  - Per-Image Metrics: count_roi, ratio_roi, percent_roi (damaged area / ROI area), density (per ROI megapixel)
  - Per-Cluster Aggregation: Median percent_roi → street_quality_roi, IQR, image count
  - Winsorization: 1%/99% tails per split independently to reduce outlier impact
  - **Variable Interpretation**: percent_roi → higher values = more detected damage = worse road condition → higher crash probability
- **Preprocessing**: Same as baseline + YOLO feature standardization
- **Performance**: Compared against baseline on train/test splits
- **Purpose**: Test if visual street damage adds predictive signal beyond infrastructure attributes

### Target Variables (Implemented)
- **Primary**: Binary crash occurrence per cluster (logistic regression)
- **Implementation**: Crash present in cluster (1) vs. no crash (0) based on point-in-polygon matching

### Feature Engineering
1. **Visual Features**: Raw detections from YOLO, aggregated per cluster
2. **Infrastructure Features**: OSM road attributes (highway type, surface, lighting, etc.)
3. **Control Variables**: road_segment_length_m, intersection features, maxspeed, lanes
4. **Standardization**: Z-score normalization for numeric features per train/val/test splits

### Model Specifications

**Baseline Logistic Regression (Implemented)**:
```python
# Features: structured road attributes only
features = ['highway', 'surface', 'lit', 'cycleway', 'sidewalk', 'oneway', 
           'bridge', 'tunnel', 'junction', 'access', 'smoothness',
           'is_intersection', 'near_traffic_signal',
           'road_segment_length_m', 'maxspeed', 'lanes', 'width', 'intersection_degree']

# Model: LogisticRegression with class_weight='balanced'
# Residuals: actual_label - predicted_probability (CNN target)
```

**Enhanced Logistic Regression with YOLO (Implemented)**:
```python
# Features: baseline + YOLO visual features
features = baseline_features + ['street_quality_roi_winz']

# YOLO Feature Extraction:
# 1. Extract first URL from list_of_thumb_1024_url per cluster
# 2. Run YOLO inference (imgsz=416, conf=0.25, iou=0.45)
# 3. Apply square ROI mask (bottom 60% height, centered horizontally)
# 4. Compute per-image metrics: count_roi, ratio_roi, percent_roi (damage/ROI), density
# 5. Aggregate per-cluster: median percent_roi → street_quality_roi
# 6. Winsorize per split: 1%/99% tails → street_quality_roi_winz
#    Higher values → more damage → worse road condition

# Model: Same LogisticRegression as baseline + YOLO feature
# Comparison: Train vs Train, Test vs Test performance
```

### Evaluation Framework
- **Metrics**: Accuracy, Precision, Recall, F1, ROC AUC, McFadden R²
- **Split Strategy**: Stratified train/val/test (70/15/15) by match_label
- **Ablation Studies**: Baseline vs. + YOLO vs. + CNN residuals
- **Marginal Gain**: Improvement from adding visual features (ΔAUC, ΔR²)

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
2. **Coverage Maps**: Spatial distribution of image counts per cluster
3. **Cluster Analysis**: Crash distribution, image statistics, temporal patterns
4. **Model Comparison**: Baseline vs YOLO vs CNN performance metrics
5. **Interpretability**: Grad-CAM visualizations for CNN predictions

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
- [x] Crash-to-image matching (exploratory 5m/10m/25m distance thresholds) (`src/features/match_crashes_to_images.py`)
- [x] Matching analysis and visualization (`src/viz/analyze_match_results.py`)
- [x] 15 m cluster dataset creation with point-in-polygon crash assignment (`src/features/create_cluster_dataset.py`)
- [x] Baseline logistic regression pipeline (`src/modeling/baseline_logistic_regression.py`)

### Completed Steps
1. **Data Collection**:
   - ✅ Query Mapillary API for Berlin imagery metadata (2013-2025, full coverage with tiling)
   - ✅ Raw data visualization (temporal and spatial analysis)
   - ✅ Download Berlin Unfallatlas crash data (2018-2024, 97,511 total crashes) and comprehensive visualization
   - ✅ Download OSM Berlin road network from Geofabrik (.pbf format)

2. **OSM Infrastructure Enrichment**:
   - ✅ Process OSM road network with pyrosm (190,231 segments, 12,204 km)
   - ✅ Match crashes to roads (91.3% match rate at 10 m threshold for OSM snapping)
   - ✅ Match images to roads (50.1% match rate at 10 m threshold for OSM snapping)
   - ✅ Generate comprehensive coverage analysis and Excel report
   - ✅ Create improved spatial visualizations (no redundancy)

3. **Crash-to-Image Matching (Exploratory)**:
   - ✅ Match crashes to images based on spatial proximity (5m, 10m, 25m thresholds for exploration)
   - ✅ Filter matches by road attributes (highway AND surface must match)
   - ✅ Generate labeled datasets with crash counts, temporal info, and shared crash tracking
   - ✅ Track crashes occurring before/after/same year as image capture
   - ✅ Analyze shared crashes (crashes matched to multiple images)
   - ✅ Output: `data/processed/mapillary_labeled_{5m,10m,25m}.csv`
   - ⚠️ **Note**: These exploratory distance-based datasets are for reference only; final predictive model uses cluster-based linking

4. **Matching Analysis & Visualization**:
   - ✅ Generate comprehensive summary statistics comparing all thresholds
   - ✅ Analyze shared crash patterns and distributions
   - ✅ Compute temporal distributions (time differences between images and crashes)
   - ✅ Create year-wise breakdowns of matched crashes
   - ✅ Generate 8 visualization types comparing thresholds
   - ✅ Output: CSV summaries and figures in `reports/Matches/`

5. **15 m Cluster Dataset Creation**:
   - ✅ Create 15 m × 15 m UTM grid clusters covering Berlin
   - ✅ Assign cluster_id to each image using spatial join
   - ✅ Aggregate images per cluster (year-independent clustering)
   - ✅ Match crashes to clusters using point-in-polygon (no distance threshold, no year filtering)
   - ✅ Track crash years as informational column (crash_years)
   - ✅ Generate cluster-level analysis reports and visualizations
   - ✅ Create stratified train/val/test split (70/15/15) by match_label
   - ✅ Output: `data/processed/clusters_with_crashes.csv` (11,567 clusters, 808 with crashes, 7.0% match rate)
   - ✅ Output: Cluster reports and figures in `reports/Clusters/`

6. **Baseline Logistic Regression Pipeline**:
   - ✅ Build interpretable baseline model predicting crash occurrence from structured road attributes
   - ✅ Feature selection: highway type, surface, lighting, cycleway, sidewalk, maxspeed, lanes, etc.
   - ✅ Exclude leakage features: coordinates, density measures, crash counts
   - ✅ Missing data handling: categorical → "missing", numeric → median from train only
   - ✅ Preprocessing pipeline: one-hot encode categoricals, pass-through numerics
   - ✅ Train logistic regression with class_weight='balanced' on train split
   - ✅ Evaluate on train/val/test splits (accuracy, precision, recall, F1, ROC AUC)
   - ✅ Compute residuals (actual - predicted probability) for CNN regression target
   - ✅ Generate comparison visualizations across splits
   - ✅ Output: `data/processed/clusters_{train,val,test}_with_residuals.csv`
   - ✅ Output: Model pipeline, metadata, coefficient report, visualizations
   - ✅ Results: ROC AUC ~0.75, high recall (0.68-0.76) but low precision (0.14) due to class imbalance

7. **CNN Residual Prediction Pipeline**:
   - ✅ Data preparation: Transform residual CSV files into per-cluster samples (one image per cluster)
   - ✅ ResNet18 architecture with regression head (predicts residual, not crash directly)
   - ✅ Two-phase training: frozen backbone (epochs 1-2) then fine-tuning (epochs 3+)
   - ✅ MPS support for Apple Silicon acceleration
   - ✅ Early stopping based on validation MSE (3 epochs without improvement)
   - ✅ Comprehensive tracking: train/val MSE and MAE per epoch
   - ✅ Grad-CAM compatibility: preserves ResNet layer structure for interpretability
   - ✅ Output: `models/cnn_residual_best.pth` (eval mode, full weights)
   - ✅ Output: `data/processed/cnn_test_predictions.csv` (image_url → predicted_residual mapping)
   - ✅ Output: `data/processed/cnn_test_top_visual_risks.csv` (ranked high-risk images)
   - ✅ Output: `reports/CNN/training_log.csv` (epoch-by-epoch metrics)
   - ✅ Output: `reports/CNN/figures/` (loss curves, scatter plots, high-risk image gallery)
   - ✅ Purpose: Learn visual safety cues that OSM attributes missed (residual = actual - predicted)

8. **Grad-CAM Interpretability Pipeline**:
   - ✅ Grad-CAM utility module with batch-safe hooks and memory management
   - ✅ Support for both Grad-CAM and Grad-CAM++ methods with flexible CLI
   - ✅ CAM intensity statistics (mean, max, top10 quantile) for correlation analysis
   - ✅ Contrast normalization and flexible colormap support (jet, turbo)
   - ✅ Stratified sampling: 5 highest, 5 lowest, 5 median, 5 random residuals
   - ✅ Automatic 3×4 panel grid generation for quartile analysis
   - ✅ Comprehensive markdown report with statistical insights and interpretations
   - ✅ Output: `reports/CNN/gradcam/` (heatmaps, overlays, grids, metadata CSV)
   - ✅ Output: `gradcam_analysis_report.md` (quartile analysis, visual patterns, bias detection)
   - ✅ Purpose: Interpret which visual features drive crash risk predictions beyond OSM attributes

9. **End-to-End Pipeline Notebook**:
   - ✅ Comprehensive Jupyter notebook reproducing full analysis pipeline
   - ✅ Notebook: `notebooks/End to End Berlin Road Crash.ipynb`
   - ✅ Covers all pipeline steps: data integration → baseline regression → YOLO features → enhanced regression → CNN → Grad-CAM
   - ✅ Smart file checking to avoid re-running long operations
   - ✅ Inline visualizations and summary statistics
   - ✅ Purpose: Reproducible end-to-end analysis workflow with clear documentation

10. **Data Validation**:
   - ✅ Verify spatial coverage and match rates
   - ✅ Check temporal consistency
   - ✅ Assess data quality and generate recommendations

### Cluster Dataset Approach
- **Cluster Size**: 15 m × 15 m squares in UTM (EPSG:25833)
- **Aggregation**: Year-independent clusters (all images in cluster aggregated together)
- **Crash Matching**: Point-in-polygon spatial matching (no distance threshold, no year filtering) - crashes assigned to clusters if they fall within cluster bounds
- **Year Information**: Tracked as informational columns (captured_at_years, crash_years)
- **Dataset Split**: Stratified by match_label to balance positive/negative examples
- **Results**: 11,567 clusters, 808 with crashes (7.0% match rate), 1,249 total matched crashes

### Implementation Workflow
The project follows a 3-stage static workflow: (1) Baseline OSM model → (2) YOLO-augmented regression → (3) CNN residual interpretation.

1. **Completed Pipeline**:
   - ✅ Created 15 m cluster grid (implemented)
   - ✅ Spatial joins: images → clusters, crashes → clusters (implemented)
   - ✅ YOLOv8 road damage detection pipeline (implemented)
   - ✅ CNN residual prediction pipeline (implemented)
   - ✅ Grad-CAM interpretability (implemented)

2. **Modeling Sequence**:
   - ✅ Baseline: Logistic regression with OSM infrastructure features
   - ✅ Enhanced: Add YOLO visual damage features (street_quality_roi_winz)
   - ✅ Residual: CNN predicts visual risk unexplained by OSM attributes

### Implementation Priorities
1. **Robustness**: Handle API rate limits and data inconsistencies
2. **Reproducibility**: Document all data sources and processing steps
3. **Scalability**: Design for efficient processing of large image datasets
4. **Validation**: Implement comprehensive quality checks

### Data Repository Note
Due to data size and licensing, raw and processed datasets are excluded from the repository. Scripts automatically download open data from Mapillary, OSM (Geofabrik), and Unfallatlas when executed.

## Risk Mitigation

### Technical Risks
- **API Rate Limits**: Implement exponential backoff and request batching
- **Data Quality**: Multiple validation layers and manual spot checks
- **Memory Management**: Process data in chunks, use efficient data types
- **Model Convergence**: Multiple random seeds, regularization

### Methodological Risks
- **Selection Bias**: Coverage controls and sensitivity analysis
- **Temporal Coverage**: Mapillary images span 2013-2025, crashes span 2018-2024; contemporaneous matching
- **Spatial Autocorrelation**: Stratified train/val/test split and clustered data consideration
- **Overfitting**: Regularization (L2) and cross-validation

## Success Criteria

### Technical Deliverables
- [x] Complete data pipeline (raw → processed)
- [x] Feature inference on 68k+ images
- [x] Cluster-level aggregated dataset
- [x] Trained models with evaluation metrics (baseline, YOLO-enhanced, CNN residuals)
- [x] Comprehensive visualization suite

### Research Deliverables
- [x] Statistical significance of visual features (YOLO: β=+0.095, p=0.031)
- [x] Marginal improvement over baseline (metrics comparison tables)
- [x] Interpretable feature importance (coefficients, Grad-CAM)
- [x] Spatial patterns in infrastructure-crash relationships (cluster analysis)

This specification serves as the comprehensive guide for executing the Berlin road infrastructure analysis project with full reproducibility and scientific rigor.
