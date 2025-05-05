"""
Visualization utilities for XAUUSD Trading Bot.

This module provides functions for generating performance charts
and visualizations for the trading bot.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import os

def plot_equity_curve(equity_curve, trades):
    """
    Plot equity curve with drawdowns.
    
    Parameters:
    -----------
    equity_curve : list
        List of account balances over time
    trades : list
        List of trade dictionaries
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot equity curve
    plt.plot(equity_curve, label='Equity')
    plt.title("Equity Curve (¢)")
    plt.ylabel("Balance (¢)")
    plt.grid(True)
    
    # Calculate and plot drawdown line
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    
    ax2 = plt.twinx()
    ax2.plot(drawdown, 'r--', alpha=0.5, label='Drawdown %')
    ax2.set_ylabel('Drawdown %')
    ax2.invert_yaxis()  # Invert to show drawdown pointing down
    
    # Combined legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Trade outcome plot
    plt.figure(figsize=(12, 4))
    profits = [t["profit"] for t in trades]
    colors = ["green" if p > 0 else "red" for p in profits]
    plt.bar(range(1, len(profits) + 1), profits, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title("Trade Outcomes (¢)")
    plt.xlabel("Trade #")
    plt.ylabel("Profit/Loss (¢)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_monthly_performance(daily_results):
    """
    Plot monthly performance heatmap.
    
    Parameters:
    -----------
    daily_results : dict
        Dictionary with dates as keys and daily profits as values
    """
    if not daily_results:
        print("No daily data available for monthly performance plot.")
        return
        
    # Group results by year and month
    monthly_data = {}
    for date, profit in daily_results.items():
        year = date.year
        month = date.month
        
        if year not in monthly_data:
            monthly_data[year] = {}
            
        if month not in monthly_data[year]:
            monthly_data[year][month] = 0
            
        monthly_data[year][month] += profit
        
    # Convert to matrix for heatmap
    years = sorted(monthly_data.keys())
    months = list(range(1, 13))
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    data_matrix = np.zeros((len(years), 12))
    
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            if month in monthly_data[year]:
                data_matrix[i, j] = monthly_data[year][month] / 100  # Convert to dollars
    
    # Create figure
    plt.figure(figsize=(12, len(years) * 0.8 + 2))
    
    # Create heatmap
    im = plt.imshow(data_matrix, cmap='RdYlGn')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Profit/Loss ($)')
    
    # Configure axis labels
    plt.xticks(np.arange(len(months)), month_names)
    plt.yticks(np.arange(len(years)), years)
    
    # Rotate month labels for better readability
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add profit values to cells
    for i in range(len(years)):
        for j in range(len(months)):
            value = data_matrix[i, j]
            text_color = 'black' if abs(value) < max(abs(data_matrix.min()), abs(data_matrix.max()))/2 else 'white'
            plt.text(j, i, f"${value:.0f}", ha="center", va="center", color=text_color)
    
    plt.title("Monthly Performance ($)")
    plt.tight_layout()
    plt.show()
    
    # Alternative: Bar chart of monthly performance
    plt.figure(figsize=(12, 6))
    
    # Convert monthly data to series
    monthly_series = {}
    for year in monthly_data:
        for month in monthly_data[year]:
            month_str = f"{year}-{month:02d}"
            monthly_series[month_str] = monthly_data[year][month] / 100  # Convert to dollars
    
    # Sort by date
    months_sorted = sorted(monthly_series.keys())
    values = [monthly_series[m] for m in months_sorted]
    
    # Plot
    colors = ['green' if v > 0 else 'red' for v in values]
    plt.bar(months_sorted, values, color=colors)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Monthly Performance ($)')
    plt.xlabel('Month')
    plt.ylabel('Profit/Loss ($)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_trade_analysis(trades, performance):
    """
    Plot trade analysis charts.
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
    performance : dict
        Performance metrics dictionary
    """
    if not trades:
        print("No trades to analyze.")
        return
    
    # Trade duration plot
    plt.figure(figsize=(12, 5))
    
    # Get duration data
    duration_data = performance['duration_distribution']
    
    categories = list(duration_data.keys())
    values = list(duration_data.values())
    
    # Create horizontal bar chart
    plt.barh(categories, values, color='skyblue')
    
    # Add values on bars
    for i, v in enumerate(values):
        plt.text(v + 0.5, i, str(v), va='center')
        
    plt.title('Trade Duration Distribution')
    plt.xlabel('Number of Trades')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Win/Loss ratio pie chart
    plt.figure(figsize=(8, 8))
    win_count = performance['winning_trades']
    loss_count = performance['losing_trades']
    
    if win_count + loss_count > 0:
        labels = [f'Wins ({win_count})', f'Losses ({loss_count})']
        sizes = [win_count, loss_count]
        colors = ['green', 'red']
        explode = (0.1, 0)  # explode the 1st slice (Wins)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
        plt.title(f"Win/Loss Ratio: {performance['win_rate']:.1f}%")
        plt.tight_layout()
        plt.show()
    
    # Profit distribution histogram
    plt.figure(figsize=(10, 6))
    profits = [t["profit"] for t in trades]
    
    # Histogram of profits
    bins = np.linspace(min(profits), max(profits), 20) if len(set(profits)) > 1 else 10
    plt.hist(profits, bins=bins, alpha=0.7, color='blue')
    plt.axvline(0, color='r', linestyle='--')  # Mark zero line
    plt.title("Profit Distribution (¢)")
    plt.xlabel("Profit/Loss (¢)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_plot(fig, filename, directory="results/charts"):
    """
    Save matplotlib figure to file.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        Base filename without extension
    directory : str
        Directory to save the chart
    """
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Full path
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    full_path = os.path.join(directory, f"{filename}_{timestamp}.png")
    
    # Save figure
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {full_path}")
    
    return full_path