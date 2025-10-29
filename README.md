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

### 5. YOLOv8 Road Damage Detection Pipeline
```bash
# Convert RDD2022 dataset to YOLO format
python src/modeling/convert_rdd_to_yolo.py

# Filter to Czech Republic + Norway only
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
│  │  ├─ tiling.py
│  │  ├─ infer_yolo.py
│  │  └─ aggregate.py
│  ├─ modeling/
│  │  ├─ prepare_datasets.py
│  │  └─ regress.py
│  └─ viz/
│     ├─ rawdata/
│     │  └─ mapillary_viz.py
│     ├─ maps.py
│     └─ plots.py
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
4. **Tiles**: create **200 m** hex or square grid in EPSG:25833 covering Berlin
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
