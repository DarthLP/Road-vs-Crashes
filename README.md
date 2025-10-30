# Berlin Road Condition → Crash Trends (2016–2024)

A end‑to‑end reproducible project that links **street‑level infrastructure quality** (from Mapillary images) to **road safety outcomes** (Berlin crash data). This project computes yearly, tile‑level visual indicators (potholes/defects, signage condition, greenery) and tests whether **changes** in those indicators predict **changes** in crash counts.

## 🔭 Project Goal

* Build a **time series** (2016–2024) of *visual infrastructure features* for Berlin using openly available street‑level imagery
* Aggregate features to **1000 m × 1000 m tiles** per year
* Merge with **annual crash counts** per tile
* Model **Δ crashes** as a function of **Δ infrastructure features**
* Deliver maps, metrics, and a brief writeup

> *Framing:* This is predictive/correlational, not causal. We control for coverage and basic network features to avoid spurious correlations.

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.10
- Conda for package management

### Grad-CAM Visualization
Generate interpretability visualizations for the CNN residual predictor:

```bash
# Activate environment
conda activate berlin-road-crash

# Generate Grad-CAM analysis (requires trained model)
python src/modeling/visualize_gradcam.py \
  --checkpoint models/cnn_residual_best.pth \
  --num_samples 20 \
  --method gradcam \
  --colormap jet \
  --save_stats

# Compare Grad-CAM vs Grad-CAM++
python src/modeling/visualize_gradcam.py \
  --checkpoint models/cnn_residual_best.pth \
  --method gradcampp \
  --output_dir reports/CNN/gradcam_plusplus
```

**Output**: `reports/CNN/gradcam/` with heatmaps, overlays, grid summaries, and analysis report.

### 📈 CNN Residual Augmentation Evaluation
Evaluate how much the CNN residual adds beyond baseline probabilities (and YOLO-enhanced):

```bash
conda activate berlin-road-crash

# Use saved predictions if present; will infer Val CNN preds if missing
python src/modeling/evaluate_cnn_augmentation.py --use_saved_preds
```

Outputs:
- `reports/CNN/cnn_augmentation_regression.csv`
- `reports/CNN/cnn_vs_baseline_threshold_metrics.csv`
- `reports/Comparison_Baseline_vs_YOLO/cnn_residual_vs_true_scatter.png`
- `reports/Comparison_Baseline_vs_YOLO/augmentation_regression_coefficients.png`
- `reports/Comparison_Baseline_vs_YOLO/delta_auc_bars.png`
- `reports/Comparison_Baseline_vs_YOLO/delta_pseudo_r2_bars.png`
- `reports/Comparison_Baseline_vs_YOLO/threshold_metrics_bars.png`

Notes:
- Thresholds are optimized on Val for F1 and Youden J; 0.5 is also reported.
- Test is used exclusively for final metrics and all figures.

### 🖼️ Image Quality Options
The project now supports multiple image resolutions from Mapillary API:
- **256px** (default): Fast processing, lower quality
- **1024px** (recommended): 4x better quality, good balance
- **2048px** (maximum): 16x better quality, best for detailed analysis

### Data Extraction Details
Current configuration extracts:
- Image quality: 1024x768 pixels (thumb_1024_url)
- Metadata: GPS coordinates, timestamp, image dimensions, compass angle, sequence ID
- Filtering: Excludes panoramic images (is_pano=True)
- Purpose: Optimized for ML-based pothole detection with YOLO models

### Crash Data Processing

The project now includes comprehensive processing and visualization of Berlin traffic accident data (2018-2021):

1. **Data Aggregation:**
   ```bash
   python src/features/aggregate_crashes.py
   ```
   - Combines 4 years of crash data (50,119 total accidents)
   - Removes unnecessary columns (LAND, LOR_ab_2021, STRASSE, etc.)
   - Standardizes coordinate formats (keeps UTM LINREFX/LINREFY)
   - Cleans data types and handles German CSV formatting

2. **Comprehensive Visualizations:**
   ```bash
   python src/viz/rawdata/crashes_viz.py
   ```
   - Temporal analysis: yearly, quarterly, monthly trends
   - Spatial distribution: crash density map on Berlin basemap
   - Severity analysis: fatal/severe/light injury distribution
   - Accident patterns: collision types, vehicle involvement
   - Temporal patterns: hour-of-day, day-of-week analysis
   - Environmental factors: lighting and road conditions
   - Advanced analysis: heatmaps and district comparisons

### 3. OSM Infrastructure Enrichment
```bash
python src/features/process_osm_efficient.py
python src/features/snap_to_roads.py
```
- Downloads Berlin OSM road network from Geofabrik (.pbf format)
- Processes road segments with infrastructure attributes
- Matches crashes and images to nearest road segments (25m threshold)
- Enriches data with road attributes: highway type, surface, maxspeed, lanes, lighting, etc.
- Saves enriched data to `data/processed/crashes_with_osm.csv` and `data/processed/mapillary_with_osm.csv`

**Key Results:**
- **Crash Match Rate**: 91.3% (88,992/97,511 crashes matched to roads)
- **Image Match Rate**: 50.1% (34,286/68,377 images matched to roads)
- **Distance Column**: `dist_to_road_m` shows distance to nearest road segment
- **Coverage**: 1.7% of images outside current OSM bounds (Brandenburg area)

### 4. Coverage Analysis & Reporting
```bash
python src/features/create_coverage_excel.py
python src/viz/improved_spatial_matching.py
python src/features/coverage_analysis.py
```
- Generates comprehensive Excel report with 7 sheets
- Creates improved spatial visualizations (no redundancy)
- Provides detailed coverage analysis and recommendations
- Saves outputs to `reports/osm_matching_coverage_analysis.xlsx` and `reports/figures/matcheddata/`

### 5. Crash-to-Image Matching
```bash
python src/features/match_crashes_to_images.py
```
- Matches crashes to Mapillary images based on spatial proximity
- Uses multiple distance thresholds (5m, 10m, 25m) for comparison
- Filters matches by road attributes (highway AND surface must match)
- Tracks temporal relationships (crashes before/after/same year as image)
- Identifies shared crashes (crashes matched to multiple images)
- Generates labeled datasets: `data/processed/mapillary_labeled_{5m,10m,25m}.csv`

**Key Results:**
- **5m threshold**: 2,063 images with crashes (6.0%), 2,865 total matches
- **10m threshold**: 5,399 images with crashes (15.7%), 9,404 total matches
- **25m threshold**: 12,267 images with crashes (35.8%), 37,493 total matches
- All images preserved in output (even those with no matches)

### 6. Matching Analysis & Visualization
```bash
python src/viz/analyze_match_results.py
```
- Generates comprehensive summary statistics comparing all thresholds
- Analyzes shared crash patterns and distributions
- Computes temporal distributions (time differences between images and crashes)
- Creates year-wise breakdowns of matched crashes
- Generates 8 visualization types comparing thresholds
- Saves CSV summaries and figures to `reports/Matches/`

**Output Files:**
- CSV summaries: matching_summary_statistics.csv, shared_crashes_statistics.csv, crash_count_distributions.csv, matched_crashes_per_year.csv, temporal_distribution.csv
- Visualizations: 8 figure files including summary dashboard, distribution comparisons, temporal analysis

### 7. Create 15m Cluster Dataset
```bash
python src/features/create_cluster_dataset.py
```
- Creates 15m × 15m spatial clusters (tiles) from Mapillary images
- Aggregates images per cluster with OSM attributes (year-independent clusters)
- Matches crashes to clusters using point-in-polygon (no distance threshold, no year filtering)
- Generates cluster-level analysis reports and visualizations
- Creates stratified train/val/test split (70/15/15) by match_label

**Key Features:**
- **Cluster Assignment**: 15m UTM grid tiles covering Berlin, assigns cluster_id to each image
- **Aggregation**: All images in a cluster aggregated together regardless of year
- **Crash Matching**: Spatial matching only (all crashes in cluster bounds), tracks crash years as info
- **Reports**: Summary statistics, crash distributions, temporal analysis, image statistics
- **Dataset Split**: Stratified by crash presence to balance positive/negative examples

**Output Files:**
- `data/processed/clusters_with_crashes.csv` - Final dataset with one row per cluster_id
- `reports/Clusters/cluster_summary_statistics.csv` - Overall cluster statistics
- `reports/Clusters/cluster_crash_distributions.csv` - Distribution of crashes per cluster
- `reports/Clusters/cluster_temporal_distribution.csv` - Crash year distribution
- `reports/Clusters/cluster_image_statistics.csv` - Image count distribution
- `reports/Clusters/figures/` - 5 visualization files including summary dashboard

### 8. Baseline Logistic Regression Pipeline (Crash Prediction)
```bash
python src/modeling/baseline_logistic_regression.py
```
- Builds interpretable baseline logistic regression model predicting crash occurrence
- Uses only structured road infrastructure attributes (no images)
- Features include: highway type, surface, lighting, cycleway, sidewalk, maxspeed, lanes, etc.
- Handles missing data appropriately (categorical → "missing", numeric → median from train)
- One-hot encodes categorical features for linear model compatibility
- Trains on train split, evaluates on train/val/test splits
- Computes residuals (actual - predicted probability) for each observation
- Residuals capture visual risk unexplained by structured data (CNN target)

**Key Results:**
- **Train metrics**: Accuracy 0.68, Precision 0.14, Recall 0.71, F1 0.24, ROC AUC 0.76
- **Val metrics**: Accuracy 0.66, Precision 0.14, Recall 0.76, F1 0.24, ROC AUC 0.75
- **Test metrics**: Accuracy 0.69, Precision 0.14, Recall 0.68, F1 0.24, ROC AUC 0.75
- Model shows high recall (catches most crashes) but low precision (many false positives)
- ROC AUC ~0.75 indicates reasonable discriminative ability despite class imbalance

**Output Files:**
- `data/processed/clusters_{train,val,test}_with_residuals.csv` - Datasets with residual column
- `models/baseline_logistic_regression.pkl` - Complete model pipeline (preprocessing + model)
- `models/baseline_logistic_regression_metadata.json` - Model metadata and hyperparameters
- `reports/baseline_logistic_regression_coefficients.csv` - Feature coefficient interpretation
- `reports/Regression_beforeCNN/baseline_metrics_comparison.csv` - Metrics across splits
- `reports/Regression_beforeCNN/` - 4 comparison visualizations (bars, lines, heatmap, individual)

### 9. YOLOv8 Road Damage Detection Pipeline
```bash
# Convert RDD2022 dataset to YOLO format
python src/modeling/convert_rdd_to_yolo.py

# Filter to Czech Republic + Norway only (train + val)
python src/modeling/filter_czech_norway.py

# Compress all images to 1024x768 (Mapillary quality)
python src/modeling/compress_cz_no_dataset.py

# Train YOLOv8s model on compressed dataset
python src/modeling/train_yolo_cz_no.py
```
- Downloads RDD2022 dataset from Kaggle using kagglehub
- Filters to Czech Republic + Norway subset (10,990 images)
- Compresses all images to 1024x768 pixels (same as Mapillary)
- Achieves 82.6% size reduction (8.4GB → 1.5GB)
- Converts annotations to YOLO format with consolidated crack classes
- Filters out drone images to focus on street-level data
- Trains YOLOv8s model to detect: crack, other corruption, Pothole
- Uses Mac-optimized MPS device for training acceleration
- Saves best weights to `runs/detect/train_cz_no/weights/best.pt`

### 9a. Test Trained YOLO Model
```bash
# Test model on test dataset
python src/modeling/test_yolo_model.py --model runs/detect/train/weights/best.pt

# Test with custom settings
python src/modeling/test_yolo_model.py \
    --model runs/detect/train/weights/best.pt \
    --imgsz 1024 \
    --conf 0.25 \
    --save-predictions \
    --num-samples 20
```
- Evaluates trained YOLOv8 model on Czech + Norway test images (1,633 images)

### 10. YOLO Street Quality Features & Enhanced Regression
```bash
# Extract YOLO features with ROI filtering
python src/modeling/extract_yolo_features.py

# Train enhanced logistic regression with YOLO features
python src/modeling/logistic_regression_with_yolo.py

# Compare baseline vs enhanced model
python src/modeling/compare_baseline_vs_yolo.py

# Run robustness analysis
python src/modeling/robustness_analysis_yolo.py
```
- Extracts robust road-focused damage metrics from YOLO detections
- Uses trapezoid ROI (bottom-centered) to reduce framing bias
- Computes per-cluster aggregated metrics: street quality, damage density, detection counts
- Winsorizes metrics per split independently (1%/99% tails)
- Trains enhanced logistic regression with YOLO features + baseline infrastructure attributes
- Compares performance against baseline model on train/test splits
- Performs robustness checks with alternative YOLO features and model specifications

**Key Features:**
- **ROI Filtering**: Trapezoid mask focuses on road area (bottom 60% of image, 70% width)
- **Per-Image Metrics**: count_roi, ratio_roi, percent_roi, density (per ROI megapixel)
- **Per-Cluster Aggregation**: Median percent_roi → street_quality_roi, IQR, image count
- **Winsorization**: Reduces outlier impact while preserving distribution shape
- **Primary Feature**: street_quality_roi_winz (winsorized street quality percentage)

**Output Files:**
- `data/processed/clusters_{train,val,test}_with_yolo_roi.csv` - Datasets with YOLO features
- `models/logistic_regression_with_yolo.pkl` - Enhanced model pipeline
- `reports/Regression_withYOLO/` - Enhanced model reports and visualizations
- `reports/Comparison_Baseline_vs_YOLO/` - Performance comparison analysis
- `reports/Regression_withYOLO/sensitivity_analysis.csv` - Robustness check results
- Computes mAP50, mAP50-95, precision, recall, F1 scores
- Generates confusion matrix and class-specific metrics
- Saves detailed evaluation reports (JSON, CSV)
- Creates visualization plots for metrics
- Optionally saves sample predictions with bounding boxes

### Setup

1. **Clone and navigate to the project:**
   ```bash
   cd Road-vs-Crashes
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate berlin-road-crash
   ```

3. **API Token Setup:**
   - The Mapillary API token is already configured in `.env` as MAPILLARY_ACCESS_TOKEN
   - To get your own token, visit: [Mapillary Developer Dashboard](https://www.mapillary.com/dashboard/developers)
   - Replace the token in `.env` if needed

4. **Install Python dependencies:**
   ```bash
   pip install python-dotenv
   ```

5. **Create directory structure:**
   ```bash
   mkdir -p data/{raw,interim,processed}
   mkdir -p notebooks
   mkdir -p src/{fetch,features,modeling,viz}
   mkdir -p reports/figures
   ```

## 📦 Expected Outputs

* `reports/figures/rawdata/images_by_year.png` (yearly image distribution)
* `reports/figures/rawdata/images_by_quarter.png` (quarterly trends 2013-2025)
* `reports/figures/rawdata/images_by_month_2018_2021.png` (monthly patterns 2018-2021)
* `reports/figures/rawdata/images_spatial_distribution.png` (spatial distribution map)
* `reports/figures/rawdata/crashes_by_year.png` (yearly crash distribution)
* `reports/figures/rawdata/crashes_by_quarter.png` (quarterly crash trends)
* `reports/figures/rawdata/crashes_by_month_2018_2021.png` (monthly crash patterns)
* `reports/figures/rawdata/crashes_spatial_distribution.png` (spatial crash distribution)
* `reports/figures/rawdata/crashes_by_severity.png` (accident severity analysis)
* `reports/figures/rawdata/crashes_by_accident_type.png` (collision type analysis)
* `reports/figures/rawdata/crashes_by_vehicle_type.png` (vehicle involvement analysis)
* `reports/figures/rawdata/crashes_by_hour.png` (hourly crash patterns)
* `reports/figures/rawdata/crashes_by_weekday.png` (day-of-week analysis)
* `reports/figures/rawdata/crashes_by_lighting.png` (lighting conditions)
* `reports/figures/rawdata/crashes_by_road_condition.png` (road conditions)
* `reports/figures/rawdata/crashes_heatmap_hour_weekday.png` (temporal heatmap)
* `reports/figures/rawdata/crashes_by_district.png` (district comparison)
* `reports/figures/berlin_dataflow.png` (dataflow diagram)
* `reports/figures/tiles_change_maps_{feature}.png` (change heatmaps)

### OSM Infrastructure Enrichment Outputs
* `reports/osm_matching_coverage_analysis.xlsx` (comprehensive Excel report with 7 sheets)
* `reports/figures/matcheddata/spatial_matching_overview.png` (geographic + distance heatmap)
* `reports/figures/matcheddata/spatial_matching_density.png` (4-panel density comparison)
* `reports/figures/matcheddata/distance_distribution_spatial.png` (color-coded distance map)
* `reports/figures/matcheddata/coverage_analysis_detailed.png` (statistical charts)
* `reports/osm_matching_coverage_report.txt` (detailed text report)
* `data/processed/crashes_with_osm.csv` (crashes enriched with road attributes)
* `data/processed/mapillary_with_osm.csv` (images enriched with road attributes)
* `data/interim/osm_berlin_roads.gpkg` (processed OSM road network)

### Crash-to-Image Matching Outputs
* `data/processed/mapillary_labeled_5m.csv` (images labeled with crash data at 5m threshold)
* `data/processed/mapillary_labeled_10m.csv` (images labeled with crash data at 10m threshold)
* `data/processed/mapillary_labeled_25m.csv` (images labeled with crash data at 25m threshold)
* `reports/Matches/matching_summary_statistics.csv` (comparison across thresholds)
* `reports/Matches/shared_crashes_statistics.csv` (shared crash analysis)
* `reports/Matches/crash_count_distributions.csv` (distribution of crash counts)
* `reports/Matches/matched_crashes_per_year.csv` (year-wise breakdown)
* `reports/Matches/temporal_distribution.csv` (time difference distributions)
* `reports/Matches/figures/` (8 visualization files including summary dashboard)

### Cluster Dataset Outputs
* `data/processed/clusters_with_crashes.csv` (15m clusters with crash matches, one row per cluster_id)
* `reports/Clusters/cluster_summary_statistics.csv` (overall cluster statistics)
* `reports/Clusters/cluster_crash_distributions.csv` (crash count distribution per cluster)
* `reports/Clusters/cluster_temporal_distribution.csv` (crash years from crash_years column)
* `reports/Clusters/cluster_image_statistics.csv` (images per cluster distribution)
* `reports/Clusters/figures/` (5 visualization files including cluster summary dashboard)

### YOLOv8 Road Damage Detection Outputs
* `data/rdd_yolo/` (YOLO format dataset with train/val splits)
* `data/rdd_yolo/data.yaml` (dataset configuration file)
* `runs/detect/train/weights/best.pt` (best trained model weights)
* `runs/detect/train/weights/last.pt` (last epoch weights)
* `runs/detect/train/predictions/` (inference test results on validation images)
* `runs/detect/train/` (training metrics, plots, and logs)

### Data Files
* `data/processed/crashes_aggregated.csv` (cleaned crash data 2018-2021)
* `data/processed/tile_year_features.parquet` (tile×year visual features)
* `data/processed/tile_year_crashes.parquet` (tile×year crash counts)
* `models/results_baseline.csv` (metrics comparing models with/without imagery)
* Short **slide deck** or **2–3 page writeup** with key findings

## 🗂️ Repository Structure

```
berlin-road-crash/
├─ data/
│  ├─ raw/               # downloaded shapefiles, Mapillary metadata (JSON/GeoJSON)
│  ├─ interim/           # filtered imagery lists, tiles
│  └─ processed/         # aggregated features, tile-year tables
├─ notebooks/
│  ├─ 00_explore.ipynb
│  ├─ 10_feature_inference.ipynb
│  └─ 20_modeling.ipynb
├─ src/
│  ├─ fetch/
│  │  ├─ mapillary_pull.py
│  │  └─ berlin_crashes_pull.py
│  ├─ features/
│  │  ├─ aggregate_crashes.py
│  │  ├─ download_osm_network.py
│  │  ├─ process_osm_efficient.py
│  │  ├─ snap_to_roads.py
│  │  ├─ match_crashes_to_images.py
│  │  ├─ create_coverage_excel.py
│  │  ├─ coverage_analysis.py
│  │  └─ create_cluster_dataset.py
│  ├─ modeling/
│  │  ├─ prepare_datasets.py
│  │  └─ regress.py
│  └─ viz/
│     ├─ rawdata/
│     │  ├─ mapillary_viz.py
│     │  ├─ crashes_viz.py
│     │  └─ combined_spatial_viz.py
│     ├─ spatial_matching_map.py
│     ├─ improved_spatial_matching.py
│     └─ analyze_match_results.py
├─ reports/
│  └─ figures/
├─ environment.yml
├─ README.md
└─ Makefile    # optional task shortcuts
```

## ⏱️ Implementation Plan

### I. Data pull & tiling
1. **Crash data (2016–2024)**: download Berlin Unfallatlas (https://unfallatlas.statistikportal.de)
2. **Mapillary metadata**: query API v4 for Berlin bounding box and years 2004–2024
3. **OSM roads**: download Berlin extract (PBF/GeoJSON)
4. **Tiles**: create **15 m** hex or square grid in EPSG:25833 covering Berlin
5. **Spatial joins**: images → tiles, crash points → tiles
6. **Coverage controls**: compute `images_per_tile_year` to control for sampling bias

### II. Visual features (inference) & aggregation
1. **Sampling strategy**: From tiles with ≥N images/year, sample up to **k=5 images** per tile per year
2. **Detector(s)**: YOLOv8s finetuned on RDD2022 dataset for road damage detection
3. **Run inference** and store per‑image predictions
4. **Aggregate features per tile×year**: mean/sum normalized by image count

### III. Modeling, evaluation, visualization
1. **Create Δ features**: `Δfeat_{t} = feat_{t} − feat_{t−1}`
2. **Crash targets**: annual counts (Poisson/nb) or rates per road length
3. **Models**: Baseline (controls only) vs. + imagery features vs. + Δ features
4. **Evaluate**: LL, AIC, RMSE on held‑out tiles (spatial CV)
5. **Visualize**: change heatmaps, partial dependence, example tiles

## 🧰 Key Dependencies

* `geopandas shapely pyproj rtree fiona`
* `requests tqdm pandas numpy`
* `ultralytics opencv-python torch torchvision`
* `rasterio earthengine-api` (optional)

## 🔌 API Usage

**Mapillary API v4 — pull image metadata for a bounding box & year**

### Quick Start with Mapillary Fetcher

1. **Get your API token:**
   - Visit: [Mapillary Developer Dashboard](https://www.mapillary.com/dashboard/developers)
   - Create a new application and copy your access token

2. **Set up environment:**
   ```bash
   export MAPILLARY_ACCESS_TOKEN='your_token_here'
   ```

3. **Run the fetcher:**
   ```bash
   conda activate berlin-road-crash
   python src/fetch/mapillary_fetch.py
   ```

### What the script does:
- ✅ Validates your API token
- ✅ Fetches images from ALL of Berlin using intelligent tiling
- ✅ Captures all available images (no year filtering)
- ✅ Saves results to `data/raw/mapillary_berlin_full.csv`
- ✅ Shows preview and summary statistics

**Coverage:** Uses 32 non-overlapping tiles to cover all of Berlin, respecting the API's 0.010 square degree limit per request. Expected to retrieve 15,000-50,000 unique images.

### Output format:
CSV with columns: `id`, `lon`, `lat`, `captured_at`, `thumb_256_url`

## 🧠 CNN Residual Prediction Pipeline

The project includes a CNN pipeline that predicts residual crash risk (unexplained by OSM attributes) from street-level images:

### 1. Data Preparation
```bash
python src/modeling/prepare_cnn_dataset.py
```
- Transforms residual CSV files into per-cluster samples
- Selects one representative image per cluster (first URL from list_of_thumb_1024_url)
- Ensures 1:1 mapping (1 cluster ⇔ 1 image ⇔ 1 residual)
- Output: `data/processed/cnn_samples_residual.csv`

### 2. CNN Training
```bash
# Normal training (26 epochs with progressive resizing)
python src/modeling/train_cnn_residual.py [--max_epochs 26 --batch_size 16]

# Resume from checkpoint (if training was interrupted)
python src/modeling/train_cnn_residual.py --resume [--max_epochs 26]
```
- ResNet18 architecture with regression head (predicts residual, not crash directly)
- **Three-phase training**: frozen backbone → fine-tuning → progressive resizing
  * Phase 1 (epochs 1-6): Frozen backbone at 224x224
  * Phase 2 (epochs 7-14): Fine-tuning at 224x224
  * Phase 3 (epochs 15-26): Progressive resizing to 424x424 for better detail
- **Persistent disk caching**: All images cached to disk on first run (much faster than URL loading)
- **Multiprocessing**: Uses 4 parallel workers for faster data loading
- Uses MPS on Apple Silicon for faster training
- **Checkpoint resuming**: Use `--resume` flag to continue from last saved checkpoint
- Early stopping based on validation MSE
- Saves model in eval mode for Grad-CAM compatibility

**Performance Optimizations:**
- **Parallel downloads**: 16 concurrent downloads for initial image caching (much faster)
- **Optimized cache**: Images resized to 224x224 and saved at 85% quality for faster loading
- **Persistent disk caching**: All images cached to `data/cache/images/` on first run
- **Subsequent runs**: Load from disk cache (100x faster than URL downloads)
- **Multiprocessing**: 4 parallel workers for data loading during training
- **Prefetching**: Prefetch 2 batches ahead for smoother training
- **Larger batch size**: Default 16 (was 8) for faster training
- **Checkpoint saving**: After every epoch for easy resuming
- **Progressive resizing**: 224x224 (fast) → 424x424 (detailed) for optimal training
- **Improved early stopping**: 8 epochs patience (was 3) for better learning

### 3. Visualization
```bash
python src/modeling/visualize_cnn_results.py
```
- Generates loss curves, scatter plots, residual distributions
- Creates gallery of top high-risk images
- Saves figures to `reports/CNN/figures/`

**Key Features:**
- **Residual Prediction**: CNN learns visual safety cues that OSM attributes missed
- **Grad-CAM Ready**: Model preserves ResNet structure for interpretability
- **One Image Per Cluster**: Clean 1:1 mapping for faster training
- **MPS Support**: Automatic GPU acceleration on Apple Silicon
- **Comprehensive Outputs**: Model weights, predictions, rankings, visualizations

**Temporal Analysis:** All images are captured regardless of date. For analysis, you can compare periods like "before 2020" vs "after 2020" using the `captured_at` field.

### Example API call (manual):
```python
import requests
import os

token = os.getenv('MAPILLARY_ACCESS_TOKEN')
url = f"https://graph.mapillary.com/images?access_token={token}&fields=id,captured_at,geometry,thumb_256_url&bbox=13.088,52.338,13.761,52.675&limit=100"
response = requests.get(url)
```

## ⚠️ Important Notes

* **Selection bias** in Mapillary (contributors favor central/wealthy areas). Mitigate with coverage controls
* **Temporal mismatch**: ensure you don't use 2024 imagery to predict 2016 crashes; always lag features
* **Non‑causal**: Do not claim that defects *cause* crashes; phrase as predictive associations

## 📚 Next Steps

1. Review the `MasterSystemPrompt.md` for detailed technical specifications
2. Set up the conda environment and install dependencies
3. Begin with Day 1 tasks: data collection and tiling
4. Follow the 3-day implementation plan systematically
