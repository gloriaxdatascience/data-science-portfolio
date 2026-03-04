import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesBuilder:
    def __init__(self, df):
        self.df = df.copy()
        self.df['published_at'] = pd.to_datetime(self.df['published_at']).dt.tz_localize(None)
        self.today = datetime(2026, 3, 1)
        
    def logistic_curve(self, t, a, b, c, d):
        """Logistic function for view accumulation"""
        return a / (1 + np.exp(-b * (t - c))) + d
    
    def build_daily_views(self, method='linear'):
        """
        Reconstruct daily view counts from current total
        Methods: 'linear', 'logistic', 'exponential'
        """
        daily_data = []
        
        for idx, video in self.df.iterrows():
            days_since = (self.today - video['published_at']).days
            
            if days_since <= 0:
                continue
                
            # Create date range
            date_range = pd.date_range(
                start=video['published_at'].date(),
                end=self.today.date(),
                freq='D'
            )
            
            if method == 'linear':
                # Simple linear assumption
                daily_views = video['view_count'] / days_since
                views_per_day = np.full(len(date_range), daily_views)
                
            elif method == 'exponential':
                # Early burst, then decay
                days = np.arange(len(date_range))
                decay_rate = 0.1
                weights = np.exp(-decay_rate * days)
                weights = weights / weights.sum()
                views_per_day = video['view_count'] * weights
                
            elif method == 'logistic':
                # S-curve: slow start, rapid growth, plateau
                days = np.arange(len(date_range))
                try:
                    # Fit logistic to approximate
                    t_norm = days / max(days)
                    popt, _ = curve_fit(
                        self.logistic_curve, 
                        t_norm, 
                        t_norm * video['view_count'],
                        p0=[video['view_count'], 5, 0.5, 0],
                        maxfev=5000
                    )
                    views_per_day = self.logistic_curve(t_norm, *popt)
                    views_per_day = np.diff(np.append([0], views_per_day))
                    views_per_day = np.maximum(views_per_day, 0)
                except:
                    # Fallback to linear
                    views_per_day = np.full(len(date_range), video['view_count'] / days_since)
            
            # Create daily records
            for i, date in enumerate(date_range):
                daily_data.append({
                    'date': date,
                    'video_id': video['video_id'],
                    'trend_name': video['trend_name'],
                    'views': views_per_day[i] if i < len(views_per_day) else 0,
                    'days_since_upload': i,
                    'title': video['title'],
                    'channel': video['channel']
                })
        
        daily_df = pd.DataFrame(daily_data)
        
        # Aggregate by trend and date
        trend_daily = daily_df.groupby(['trend_name', 'date'])['views'].sum().reset_index()
        
        # Add rolling averages
        for trend in trend_daily['trend_name'].unique():
            mask = trend_daily['trend_name'] == trend
            trend_daily.loc[mask, 'views_smooth_7d'] = (
                trend_daily.loc[mask, 'views'].rolling(7, min_periods=1).mean()
            )
            trend_daily.loc[mask, 'views_smooth_30d'] = (
                trend_daily.loc[mask, 'views'].rolling(30, min_periods=1).mean()
            )
        
        return trend_daily
    
    def calculate_velocity(self, daily_df):
        """Calculate view velocity (first derivative)"""
        df = daily_df.copy()
        
        # Daily change
        df['views_velocity'] = df.groupby('trend_name')['views'].diff()
        
        # Acceleration (second derivative)
        df['views_acceleration'] = df.groupby('trend_name')['views_velocity'].diff()
        
        return df
    
    def detect_peaks(self, daily_df, window=7):
        """Find peak dates for each trend"""
        peaks = []
        
        for trend in daily_df['trend_name'].unique():
            trend_data = daily_df[daily_df['trend_name'] == trend].copy()
            
            # Find maximum
            peak_idx = trend_data['views_smooth_7d'].idxmax()
            peak_row = trend_data.loc[peak_idx]
            
            # Calculate half-life (time to drop to 50% of peak)
            peak_value = peak_row['views_smooth_7d']
            half_value = peak_value / 2
            
            # Find first date after peak where views drop below half
            post_peak = trend_data[trend_data['date'] > peak_row['date']]
            half_life_days = None
            
            if not post_peak.empty:
                below_half = post_peak[post_peak['views_smooth_7d'] <= half_value]
                if not below_half.empty:
                    half_life_days = (below_half.iloc[0]['date'] - peak_row['date']).days
            
            peaks.append({
                'trend_name': trend,
                'peak_date': peak_row['date'],
                'peak_views': peak_value,
                'total_views': trend_data['views'].sum(),
                'half_life_days': half_life_days,
                'days_to_peak': (peak_row['date'] - trend_data['date'].min()).days
            })
        
        return pd.DataFrame(peaks)

# Usage example
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv('data/raw/all_trends_complete.csv')
    
    # Build time series
    builder = TimeSeriesBuilder(df)
    
    ## Try different methods
    #daily_linear = builder.build_daily_views(method='linear')
    #daily_logistic = builder.build_daily_views(method='logistic')
    #
    ## Add velocity
    #daily_with_velocity = builder.calculate_velocity(daily_linear)

    # Switch from linear to exponential
    daily_exp = builder.build_daily_views(method='exponential')
    daily_exp = builder.calculate_velocity(daily_exp)
    
    # Detect peaks
    #peaks_df = builder.detect_peaks(daily_linear)
    peaks_df = builder.detect_peaks(daily_exp)
    print("\n📈 Peak Analysis:")
    print(peaks_df)
    
    # Save
    #daily_linear.to_csv('data/processed/daily_views_linear.csv', index=False)
    daily_exp.to_csv('data/processed/daily_views_exponential.csv', index=False)
    peaks_df.to_csv('data/processed/peak_analysis.csv', index=False)