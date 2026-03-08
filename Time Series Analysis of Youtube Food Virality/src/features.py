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
    
    def analyze_keyword_impact(self):
        """Analyze keyword frequency and impact from video titles"""
        import re

        # These are words that appear in titles but tell us nothing trend-specific
        STOPWORDS = {
            'a','an','the','and','or','in','on','at','to','for','of','is','it','its',
            'this','that','with','how','make','making','made','i','my','you','your',
            'me','we','our','are','was','be','do','did','get','got','can','will',
            'video','youtube','shorts','viral','food','recipe','recipes','cooking',
            'cook','cooked','trying','tried','try','eat','eating','ate','taste',
            'tasting','watch','new','best','good','real','just','so','the','what',
            'more','all','its','but','not','have','has','from','out','one','top',
            'first','last','ever'
        }

        rows = []
        for _, row in self.raw_df.iterrows():
            text = str(row.get('title', ''))
            tokens = re.findall(r"[a-z]{3,}", text.lower())
            keywords = [t for t in tokens if t not in STOPWORDS]

            for kw in set(keywords):  # set = count each keyword once per video
                rows.append({
                    'keyword': kw,
                    'trend': row['trend_name'],
                    'like_count': row['like_count'],
                    'comment_count': row['comment_count']
                })

        kw_df = pd.DataFrame(rows)

        keyword_stats = []
        for kw, group in kw_df.groupby('keyword'):
            if len(group) < 2:  # skip keywords appearing in only 1 video
                continue

            relevant_trends = group['trend'].unique()
            per_trend_peaks = []
            per_trend_half_lives = []

            for trend in relevant_trends:
                trend_daily = self.daily_df[self.daily_df['trend_name'] == trend]
                if trend_daily.empty:
                    continue
                video_count = len(group[group['trend'] == trend])
                trend_peak = trend_daily['views'].max()
                trend_half_life = len(trend_daily[trend_daily['views'] >= (trend_peak / 2)])

                per_trend_peaks.extend([trend_peak] * video_count)
                per_trend_half_lives.extend([trend_half_life] * video_count)

            if not per_trend_peaks:
                continue

            keyword_stats.append({
                'keyword': kw,
                'frequency': len(group),
                'trend_count': len(relevant_trends),
                'peak_views': np.mean(per_trend_peaks),
                'half_life': np.mean(per_trend_half_lives),
                'total_engagement': group[['like_count', 'comment_count']].sum().sum()
            })

        return pd.DataFrame(keyword_stats)    
    
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
    
    # Analyze keywords
    keyword_impact = engineer.analyze_keyword_impact()
    print("\n🔑 Trend-Specific Keywords (appear in only 1 trend):")
    print(keyword_impact[keyword_impact['trend_count'] == 1]
        .sort_values('peak_views', ascending=False)
        .head(10)[['keyword', 'trend_count', 'peak_views', 'half_life']]
        .round(1).to_string())
    
    print("\n🔑 Cross-Trend Keywords (appear in 2+ trends):")
    print(keyword_impact[keyword_impact['trend_count'] > 1]
        .sort_values('frequency', ascending=False)
        .head(10)[['keyword', 'trend_count', 'frequency', 'peak_views', 'half_life']]
        .round(1).to_string())
    
    # Save features
    features_df.to_csv('data/processed/trend_features.csv', index=False)
    keyword_impact.to_csv('data/processed/keyword_impact.csv', index=False)