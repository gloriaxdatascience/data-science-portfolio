import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates

class TrendVisualizer:
    def __init__(self, daily_df, peaks_df):
        self.daily_df = daily_df
        self.peaks_df = peaks_df
        
    def plot_all_trends_overlay(self, save_path='data/figures/all_trends_overlay.png'):
        """The money chart - all trends on one plot"""
        plt.figure(figsize=(14, 8))
        
        # Normalize each trend to its peak for comparison
        for trend in self.daily_df['trend_name'].unique():
            trend_data = self.daily_df[self.daily_df['trend_name'] == trend].copy()
            peak = trend_data['views_smooth_7d'].max()
            if peak > 0:
                trend_data['normalized'] = trend_data['views_smooth_7d'] / peak
                
                # Align to peak date
                peak_date = self.peaks_df[self.peaks_df['trend_name'] == trend]['peak_date'].values[0]
                trend_data['days_from_peak'] = (trend_data['date'] - pd.to_datetime(peak_date)).dt.days
                
                # Plot only -60 to +60 days around peak
                plot_data = trend_data[abs(trend_data['days_from_peak']) <= 60]
                plt.plot(plot_data['days_from_peak'], plot_data['normalized'], 
                        linewidth=2, label=trend, alpha=0.8)
        
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.3, label='Peak')
        plt.xlabel('Days from Peak', fontsize=12)
        plt.ylabel('Normalized Views (7-day avg)', fontsize=12)
        plt.title('Viral Cooking Trends: Rise and Fall Patterns', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_trend_lifecycles(self, save_path='data/figures/lifecycle_subplots.png'):
        """Individual subplots for each trend"""
        trends = self.daily_df['trend_name'].unique()
        n_trends = len(trends)
        n_cols = 2
        n_rows = (n_trends + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
        
        for i, trend in enumerate(trends):
            trend_data = self.daily_df[self.daily_df['trend_name'] == trend].copy()
            peak_info = self.peaks_df[self.peaks_df['trend_name'] == trend].iloc[0]
            
            ax = axes[i]
            
            # Plot raw and smoothed
            ax.plot(trend_data['date'], trend_data['views'], alpha=0.3, color='gray', linewidth=1)
            ax.plot(trend_data['date'], trend_data['views_smooth_7d'], linewidth=2, color='blue')
            
            # Mark peak
            ax.axvline(x=pd.to_datetime(peak_info['peak_date']), color='red', linestyle='--', alpha=0.7)
            
            # Add half-life annotation
            if pd.notna(peak_info['half_life_days']):
                ax.annotate(f"Half-life: {peak_info['half_life_days']} days", 
                           xy=(0.7, 0.9), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
            
            # Formatting
            ax.set_title(f"{trend} (Peak: {peak_info['peak_date'].date()})", fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Views (7-day avg)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, save_path='data/figures/dashboard.html'):
        """Interactive Plotly dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Normalized Trends', 'Peak Views Comparison', 
                          'Half-Life Analysis', 'Views Distribution',
                          'Velocity Heatmap', 'Trend Characteristics'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'box'}],
                   [{'type': 'heatmap'}, {'type': 'table'}]]
        )
        
        # 1. Normalized trends
        for trend in self.daily_df['trend_name'].unique():
            trend_data = self.daily_df[self.daily_df['trend_name'] == trend].copy()
            peak = trend_data['views_smooth_7d'].max()
            if peak > 0:
                trend_data['normalized'] = trend_data['views_smooth_7d'] / peak
                fig.add_trace(
                    go.Scatter(x=trend_data['date'], y=trend_data['normalized'],
                              name=trend, mode='lines'),
                    row=1, col=1
                )
        
        # 2. Peak views
        fig.add_trace(
            go.Bar(x=self.peaks_df['trend_name'], y=self.peaks_df['peak_views'],
                  name='Peak Views'),
            row=1, col=2
        )
        
        # 3. Half-life
        fig.add_trace(
            go.Bar(x=self.peaks_df['trend_name'], y=self.peaks_df['half_life_days'],
                  name='Half-Life (days)'),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(height=1200, showlegend=False,
                         title_text="Viral Cooking Trends Dashboard")
        
        fig.write_html(save_path)
        return fig
    
    def plot_velocity_heatmap(self, save_path='data/figures/velocity_heatmap.png'):
        """Heatmap of view velocity over time"""
        # Pivot data for heatmap
        pivot = self.daily_df.pivot_table(
            values='views_velocity',
            index='trend_name',
            columns=self.daily_df['date'].dt.to_period('M'),
            aggfunc='mean'
        )
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot, cmap='RdYlGn', center=0, annot=False, cbar_kws={'label': 'View Velocity'})
        plt.title('View Velocity Heatmap: When Trends are Growing Fastest', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Trend')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Usage
if __name__ == "__main__":
    daily_df = pd.read_csv('data/processed/daily_views_exponential.csv')
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    peaks_df = pd.read_csv('data/processed/peak_analysis.csv')
    peaks_df['peak_date'] = pd.to_datetime(peaks_df['peak_date'])
    
    daily_df = daily_df.sort_values(['trend_name', 'date'])
    daily_df['views_velocity'] = daily_df.groupby('trend_name')['views'].diff()

    viz = TrendVisualizer(daily_df, peaks_df)
    
    # Generate all visualizations
    viz.plot_all_trends_overlay()
    viz.plot_trend_lifecycles()
    viz.create_interactive_dashboard()
    viz.plot_velocity_heatmap()