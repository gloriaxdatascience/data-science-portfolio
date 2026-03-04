import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self, daily_df, peaks_df, raw_df):
        self.daily_df = daily_df
        self.peaks_df = peaks_df
        self.raw_df = raw_df
        
    def create_trend_features(self):
        """Create feature matrix for each trend"""
        features = []
        
        for trend in self.daily_df['trend_name'].unique():
            trend_daily = self.daily_df[self.daily_df['trend_name'] == trend]
            trend_raw = self.raw_df[self.raw_df['trend_name'] == trend]
            peak_info = self.peaks_df[self.peaks_df['trend_name'] == trend].iloc[0]
            
            # Time-based features
            total_days = (trend_daily['date'].max() - trend_daily['date'].min()).days
            days_to_peak = peak_info['days_to_peak']
            
            # Volume features
            total_views = trend_daily['views'].sum()
            peak_views = peak_info['peak_views']
            avg_daily_views = trend_daily['views'].mean()
            std_daily_views = trend_daily['views'].std()
            
            # Growth features
            pre_peak = trend_daily[trend_daily['date'] <= peak_info['peak_date']]
            post_peak = trend_daily[trend_daily['date'] > peak_info['peak_date']]
            
            if len(pre_peak) > 1:
                growth_rate = (pre_peak['views'].iloc[-1] - pre_peak['views'].iloc[0]) / len(pre_peak)
            else:
                growth_rate = 0
                
            if len(post_peak) > 1:
                decay_rate = (post_peak['views'].iloc[0] - post_peak['views'].iloc[-1]) / len(post_peak)
            else:
                decay_rate = 0
            
            # Video-level features
            avg_videos_per_week = len(trend_raw) / (total_days / 7)
            avg_likes_per_video = trend_raw['like_count'].mean()
            avg_comments_per_video = trend_raw['comment_count'].mean()
            
            # Engagement ratio
            engagement_rate = (avg_likes_per_video + avg_comments_per_video) / avg_daily_views if avg_daily_views > 0 else 0
            
            # Tags analysis
            all_tags = []
            for tags in trend_raw['tags'].dropna():
                if isinstance(tags, str):
                    all_tags.extend(eval(tags) if tags.startswith('[') else [tags])
            
            unique_tags = len(set(all_tags))
            top_tags = pd.Series(all_tags).value_counts().head(5).to_dict() if all_tags else {}
            
            # Shape classification
            if days_to_peak < 30 and peak_info['half_life_days'] < 30:
                shape = "sharp spike"
            elif days_to_peak < 30 and peak_info['half_life_days'] >= 30:
                shape = "spike with long tail"
            elif days_to_peak >= 30 and peak_info['half_life_days'] < 30:
                shape = "slow rise, fast fall"
            else:
                shape = "slow burn"
            
            features.append({
                'trend_name': trend,
                'total_views': total_views,
                'peak_views': peak_views,
                'avg_daily_views': avg_daily_views,
                'std_daily_views': std_daily_views,
                'total_days': total_days,
                'days_to_peak': days_to_peak,
                'half_life_days': peak_info['half_life_days'],
                'growth_rate': growth_rate,
                'decay_rate': decay_rate,
                'growth_to_decay_ratio': growth_rate / decay_rate if decay_rate != 0 else np.inf,
                'avg_videos_per_week': avg_videos_per_week,
                'avg_likes_per_video': avg_likes_per_video,
                'avg_comments_per_video': avg_comments_per_video,
                'engagement_rate': engagement_rate,
                'unique_tags': unique_tags,
                'top_tags': str(top_tags),
                'shape_classification': shape,
                'peak_intensity': peak_views / avg_daily_views,
                'coefficient_variation': std_daily_views / avg_daily_views if avg_daily_views > 0 else 0
            })
        
        return pd.DataFrame(features)
    
    def analyze_tag_impact(self):
        """Correlate tags with trend characteristics"""
        tag_impact = []
        
        for trend in self.raw_df['trend_name'].unique():
            trend_raw = self.raw_df[self.raw_df['trend_name'] == trend]
            trend_peak = self.peaks_df[self.peaks_df['trend_name'] == trend].iloc[0]
            
            # Extract all tags for this trend
            all_tags = []
            for tags in trend_raw['tags'].dropna():
                if isinstance(tags, str):
                    try:
                        tag_list = eval(tags) if tags.startswith('[') else [tags]
                        all_tags.extend(tag_list)
                    except:
                        all_tags.append(tags)
            
            # Count tag frequency
            tag_counts = pd.Series(all_tags).value_counts()
            
            # For top tags, record impact
            for tag, count in tag_counts.head(10).items():
                tag_impact.append({
                    'tag': tag,
                    'trend': trend,
                    'frequency': count,
                    'peak_views': trend_peak['peak_views'],
                    'half_life': trend_peak['half_life_days'],
                    'days_to_peak': trend_peak['days_to_peak']
                })
        
        return pd.DataFrame(tag_impact)
    
    def classify_trend_shape(self):
        """Cluster trends by shape"""
        from sklearn.cluster import KMeans
        
        # Create feature matrix for clustering
        feature_cols = ['days_to_peak', 'half_life_days', 'peak_intensity', 'coefficient_variation']
        
        # Prepare data
        X = self.create_trend_features()[feature_cols].fillna(0)
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Map clusters to shapes
        shape_map = {
            0: "Early Peak, Fast Decay",
            1: "Late Peak, Sustained",
            2: "Moderate Rise and Fall"
        }
        
        return clusters, shape_map

# Usage
if __name__ == "__main__":
    daily_df = pd.read_csv('data/processed/daily_views_exponential.csv')
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    peaks_df = pd.read_csv('data/processed/peak_analysis.csv')
    peaks_df['peak_date'] = pd.to_datetime(peaks_df['peak_date'])
    raw_df = pd.read_csv('data/raw/all_trends_complete.csv')
    
    engineer = FeatureEngineer(daily_df, peaks_df, raw_df)
    
    # Create features
    features_df = engineer.create_trend_features()
    print("\n📊 Trend Features:")
    print(features_df[['trend_name', 'shape_classification', 'peak_intensity', 
                       'growth_to_decay_ratio']].to_string())
    
    # Analyze tags
    tag_impact = engineer.analyze_tag_impact()
    print("\n🏷️ Top Tags Impact:")
    print(tag_impact.groupby('tag').agg({
        'peak_views': 'mean',
        'half_life': 'mean'
    }).sort_values('peak_views', ascending=False).head(10))
    
    # Save features
    features_df.to_csv('data/processed/trend_features.csv', index=False)
    tag_impact.to_csv('data/processed/tag_impact.csv', index=False)