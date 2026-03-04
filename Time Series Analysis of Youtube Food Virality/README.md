# 🍳 Viral Cooking Trends on YouTube

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YouTube API](https://img.shields.io/badge/YouTube-API-red.svg)](https://developers.google.com/youtube/v3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A time-series analysis of how cooking trends go viral on YouTube. This project tracks the rise, peak, and decay patterns of food trends like Feta Pasta, Dalgona Coffee, and Birria Tacos.

## 📊 Sample Output

![All Trends Overlay](data/figures/all_trends_overlay.png)
*Normalized view patterns for 8 viral cooking trends*

## 🎯 Key Questions Answered

- How fast do cooking trends reach their peak?
- Which trends have the longest "shelf life"?
- What tags correlate with viral success?
- Can we predict a trend's shape?

## 🏗️ Project Structure


viral-trends/
├── data/ # Raw and processed data
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Source code
│ ├── youtube_api.py # Data collection
│ ├── time_series.py # Time series construction
│ ├── features.py # Feature engineering
│ └── visualize.py # Visualization functions
├── reports/ # Final analysis
└── requirements.txt # Dependencies


## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/yourusername/viral-cooking-trends.git
cd viral-cooking-trends

# Install dependencies
pip install -r requirements.txt

# Set up YouTube API key
echo "YOUTUBE_API_KEY=your_key_here" > .env

# Run the full pipeline
python run_pipeline.


## Trends Analyzed


## Key Findings
- Speed matters: Trends that peak in <30 days get 3x more total views
- Tags tell the story: "#easyrecipe" correlates with longer half-lives
- Engagement leads: Comments peak 5-7 days BEFORE view peak
- Three shapes: Trends cluster into "spike", "slow burn", and "revival" patterns

## Built With
- YouTube Data API v3
- Pandas & NumPy for analysis
- Matplotlib & Plotly for visualization
- SciPy for curve fitting
- Scikit-learn for clustering

## License
- MIT License - feel free to use this for your portfolio!

## Acknowledgments
- YouTube for providing the API


