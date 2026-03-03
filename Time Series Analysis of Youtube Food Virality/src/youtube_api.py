import os
import time
import pandas as pd
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import yaml

load_dotenv()

class YouTubeDataCollector:
    def __init__(self):
        self.api_key = os.getenv('YOUTUBE_API_KEY')
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
    def search_videos(self, query, max_results=50):
        """Search for videos by keyword"""
        videos = []
        next_page_token = None
        
        try:
            while len(videos) < max_results:
                request = self.youtube.search().list(
                    q=query,
                    part='snippet',
                    type='video',
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token,
                    order='relevance'  # or 'viewCount' or 'date'
                )
                response = request.execute()
                
                # Extract video IDs
                video_ids = [item['id']['videoId'] for item in response['items']]
                
                # Get detailed stats for these videos
                stats = self.get_video_details(video_ids)
                videos.extend(stats)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                time.sleep(0.5)  # Be nice to the API
                
        except HttpError as e:
            print(f"API Error: {e}")
            
        return videos[:max_results]
    
    def get_video_details(self, video_ids):
        """Get statistics for specific videos"""
        if not video_ids:
            return []
            
        request = self.youtube.videos().list(
            part='snippet,statistics',
            id=','.join(video_ids)
        )
        response = request.execute()
        
        videos = []
        for item in response['items']:
            video = {
                'video_id': item['id'],
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'tags': item['snippet'].get('tags', []),
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'comment_count': int(item['statistics'].get('commentCount', 0)),
                'category': self.get_category_name(item['snippet'].get('categoryId', '0'))
            }
            videos.append(video)
            
        return videos
    
    def get_category_name(self, category_id):
        """Convert category ID to name"""
        # Cooking/food is usually category 26 (Howto & Style) or 20 (Gaming? no)
        # Let's map common ones
        categories = {
            '26': 'Howto & Style',  # Most cooking videos
            '10': 'Music',  # Rare for food
            '20': 'Gaming',  # Nope
            '23': 'Comedy',  # Some food comedy
        }
        return categories.get(category_id, 'Other')
    
    def collect_trend_data(self, config_path='config/trends.yaml'):
        """Main collection function"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        all_data = []
        for trend in config['trends']:
            print(f"\n📊 Collecting: {trend['name']}")
            
            for keyword in trend['keywords']:
                print(f"  🔍 Searching: '{keyword}'")
                videos = self.search_videos(keyword, max_results=25)
                
                for video in videos:
                    video['trend_name'] = trend['name']
                    video['search_keyword'] = keyword
                    all_data.append(video)
                
                time.sleep(1)  # Rate limiting
            
            # Save per trend as backup
            trend_df = pd.DataFrame([v for v in all_data if v['trend_name'] == trend['name']])
            trend_df.to_csv(f'data/raw/{trend["name"].replace(" ", "_")}_raw.csv', index=False)
        
        return pd.DataFrame(all_data)

if __name__ == "__main__":
    collector = YouTubeDataCollector()
    
    # Test with one trend first
    test_videos = collector.search_videos("crookie recipe", max_results=5)
    print(f"Found {len(test_videos)} videos")
    
    # Then run full collection
    # df = collector.collect_trend_data()
    # df.to_csv('data/raw/all_trends_raw.csv', index=False)