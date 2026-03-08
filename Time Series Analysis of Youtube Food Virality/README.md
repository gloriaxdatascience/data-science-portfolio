# Viral Cooking Trends on YouTube  
Time Series Analysis of Rise, Peak, and Decay Patterns

This project analyzes how viral cooking trends spread on YouTube, how quickly they peak, and how long they remain relevant. Using time series modeling, feature engineering, and engagement analysis, the project uncovers lifecycle patterns behind trends like *crookie*, *dirty bread*, *tanghulu*, and more.

---

## 📂 Project Structure

project/
│
├── data/
│   ├── raw/                # Raw YouTube API data
│   ├── processed/          # Cleaned & engineered datasets
│   └── figures/            # Generated plots
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_final_analysis.ipynb   # Main report notebook
│
├── src/
│   ├── youtube_api.py      # Data collection
│   ├── time_series.py      # Daily views + smoothing
│   ├── features.py         # Feature engineering
│   └── visualize.py        # Plotting utilities
│
├── reports/
│   └── final_analysis.md   # Summary report
│
├── run_pipeline.py         # Optional end‑to‑end pipeline
└── requirements.txt

---

## 🎯 Goals

- Understand how viral cooking trends evolve over time  
- Identify rise speed, peak intensity, and decay rate  
- Compare lifecycle shapes across trends  
- Analyze engagement patterns (likes, comments, velocity)  
- Extract keyword signals that drive virality  

---

## 📊 Key Findings

### ⚡ Fastest to Peak  
**dubai chocolate** — reached peak in **150 days**, with **10.9×** average intensity.

### 🐢 Longest Lasting  
**chopped italian sandwich** — half‑life **21 days**, total active **334 days**.

### 🔥 Most Intense Peak  
**dirty bread** — **158×** average daily views, **240M** total views.

### ❤️ Most Engaging  
**crookie** — engagement rate **1.77**, avg **99k likes/video**.

### 📈 Shape Classification  
All trends in this dataset follow a **slow rise, fast fall** pattern.

---

## 🧪 Methods

### Time Series Modeling
- Daily view aggregation  
- 7‑day exponential smoothing  
- Peak detection  
- Velocity and acceleration metrics  

### Feature Engineering
- Days to peak  
- Half-life estimation  
- Peak intensity  
- Total views  
- Engagement rate  

### Keyword Analysis
- Tokenization of video titles  
- Stopword filtering  
- Keyword → trend mapping  
- Engagement-weighted ranking  

---

## 📷 Visualizations

The project generates:

- Trend lifespans  
- Shape clusters  
- Keyword impact charts  
- Engagement lifecycle curves  

All figures are saved to:

data/figures/

---

## ▶️ Running the Project

### Notebook workflow
Open:

notebooks/00_analysis.ipynb

This runs the full analysis end‑to‑end.

---

## 📦 Installation

pip install -r requirements.txt


---

## 📝 Report

A final summary is available at:

reports/final_analysis.md
