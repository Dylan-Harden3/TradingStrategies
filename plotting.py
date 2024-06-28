import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_entries_per_year(stock_df):
    df = stock_df.copy()
    df['Year'] = df['Date'].dt.year

    rows_per_year = df.groupby('Year').size()

    rows_per_year = rows_per_year.reset_index(name='Count')

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Year', y='Count', data=rows_per_year, palette='viridis')
    plt.title('Number of Rows per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Rows')
    plt.xticks(rotation=60)
    plt.show()

def plot_returns(dates, daily_returns, cumulative_returns, portfolio_values, metrics):
    fig, axs = plt.subplots(2, 1, figsize=(14, 18))
    
    # Plot 1: Cumulative Returns and Portfolio Value
    ax1 = axs[0]
    ax1.plot(dates, cumulative_returns, label='Cumulative Returns', color='blue')
    ax1.set_ylabel('Cumulative Return', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, portfolio_values, label='Portfolio Value', color='green')
    ax1_twin.set_ylabel('Portfolio Value', color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    
    ax1.set_title('Cumulative Returns and Portfolio Value Over Time')
    ax1.grid(True)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Plot 2: Daily Returns
    ax2 = axs[1]
    ax2.plot(dates, daily_returns, label='Daily Returns', color='orange')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_ylabel('Daily Return')
    ax2.set_title('Daily Returns Over Time')
    ax2.grid(True)
    ax2.legend()

    # Add text box with metrics
    textstr = '\n'.join((
        f"Total Return: {metrics['Total Return']:.2%}",
        f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}",
        f"Win Rate: {metrics['Win Rate']:.2%}",
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.05, 0.95, textstr, transform=axs[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

def plot_comparative_returns(results):
    fig, axs = plt.subplots(2, 1, figsize=(14, 18))
    
    colors = ['blue', 'red']
    
    # Plot 1: Cumulative Returns
    ax1 = axs[0]
    for (strategy_name, data), color in zip(results.items(), colors):
        ax1.plot(data['dates'], data['cumulative_returns'], label=f'{strategy_name} Cumulative Returns', color=color)
    
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title('Cumulative Returns Over Time')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Daily Returns
    ax2 = axs[1]
    for (strategy_name, data), color in zip(results.items(), colors):
        ax2.plot(data['dates'], data['daily_returns'], label=f'{strategy_name} Daily Returns', color=color, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_ylabel('Daily Return')
    ax2.set_title('Daily Returns Over Time')
    ax2.grid(True)
    ax2.legend()

    # Add text box with metrics
    textstr = []
    for strategy_name, data in results.items():
        metrics = data['metrics']
        textstr.append('\n'.join((
            f"{strategy_name}:",
            f"Total Return: {metrics['Total Return']:.2%}",
            f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}",
            f"Win Rate: {metrics['Win Rate']:.2%}"
        )))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.05, 0.95, '\n\n'.join(textstr), transform=axs[0].transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

def print_comparative_metrics(results):
    for strategy_name, data in results.items():
        print(f"\nMetrics for {strategy_name}:")
        for key, value in data['metrics'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
