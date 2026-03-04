# quick_test.py
from .youtube_api import YouTubeDataCollector
import pandas as pd

collector = YouTubeDataCollector()

# Test a single trend
print("Testing with 'tanghulu'...")
videos = collector.search_videos("tanghulu recipe", max_results=10)
df_test = pd.DataFrame(videos)
print(f"Collected {len(df_test)} videos")
df_test.to_csv('data/raw/test_tanghulu.csv', index=False)

# Now run full collection
print("\nRunning full collection for all trends...")
df_all = collector.collect_trend_data()
df_all.to_csv('data/raw/all_trends_complete.csv', index=False)

print(f"\nTotal videos collected: {len(df_all)}")
print(f"Trends covered: {df_all['trend_name'].nunique()}")