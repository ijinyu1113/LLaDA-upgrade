import pandas as pd
import matplotlib.pyplot as plt

def plot_ala_results(csv_path='./ala_results/rank_comparison.csv'):
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 2. Plot Loss on the left axis
    color_loss = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Cross-Entropy Loss', color=color_loss, fontsize=12)
    ax1.plot(df['step'], df['loss'], color=color_loss, label='Training Loss', linewidth=2, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(alpha=0.3)

    # 3. Create a second axis for Rank
    ax2 = ax1.twinx()
    color_rank = 'tab:blue'
    ax2.set_ylabel('Effective Rank (Isotropy)', color=color_rank, fontsize=12)
    
    # Plot Baseline vs ALA
    ax2.plot(df['step'], df['baseline_rank'], color='gray', linestyle='--', label='LLaDA Baseline Rank', alpha=0.6)
    ax2.plot(df['step'], df['ala_rank'], color=color_rank, label='ALA Router Rank', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_rank)

    # 4. Formatting
    plt.title('The "Symmetry Trap": Loss vs. Representation Collapse', fontsize=14, pad=15)
    fig.tight_layout()
    
    # Combined Legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.savefig('./ala_results/rank_collapse_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to ./ala_results/rank_collapse_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_ala_results()