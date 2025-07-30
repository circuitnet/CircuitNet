import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_log_file(file_path):
    """
    Parse log file to extract iteration numbers and loss values
    """
    iterations = []
    losses = []
    
    # Pattern to match the specified format
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ===> Iters\[(\d+)\]\(\d+/\d+\): Avg Loss: ([\d.]+), Current LR: [\d.e-]+'
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(pattern, line.strip())
                if match:
                    iteration = int(match.group(1))
                    loss = float(match.group(2))
                    iterations.append(iteration)
                    losses.append(loss)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], []
    
    return iterations, losses

def create_loss_curve(iterations, losses, output_path, title):
    """
    Create and save loss curve plot with three y-axes
    """
    if not iterations or not losses:
        print(f"No valid data found for {title}")
        return
    
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot original loss on left y-axis
    color1 = 'tab:blue'
    ax1.set_xlabel('Iterations', fontsize=12)
    ax1.set_ylabel('Average Loss (Original)', fontsize=12, color=color1)
    line1 = ax1.plot(iterations, losses, color=color1, linewidth=1, alpha=0.7, label='Original')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Create second y-axis for smoothed data
    ax2 = ax1.twinx()
    
    # Add smoothed line if there are enough points
    if len(iterations) > 10:
        # Calculate moving average for smoothing
        window_size = min(50, len(losses) // 10)
        if window_size > 1:
            smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_iterations = iterations[window_size-1:]
            
            color2 = 'tab:red'
            ax2.set_ylabel(f'Average Loss (Smoothed, window={window_size})', fontsize=12, color=color2)
            line2 = ax2.plot(smoothed_iterations, smoothed_losses, color=color2, linewidth=2, 
                            alpha=0.8, label=f'Smoothed (window={window_size})')
            ax2.tick_params(axis='y', labelcolor=color2)
        else:
            line2 = []
    else:
        line2 = []
    
    # Create third y-axis for log scale data
    ax3 = ax1.twinx()
    # Offset the third axis to the right
    ax3.spines['right'].set_position(('outward', 60))
    
    color3 = 'tab:green'
    ax3.set_ylabel('Average Loss (Log Scale)', fontsize=12, color=color3)
    # Use log scale for losses, but avoid log(0) by adding small epsilon
    log_losses = np.log(np.array(losses) + 1e-10)
    line3 = ax3.plot(iterations, log_losses, color=color3, linewidth=1.5, 
                     alpha=0.8, label='Log Scale', linestyle='--')
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Combine legends from all axes
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.title(f'Training Loss Curve - {title}', fontsize=14)
    
    # Format axes
    ax1.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    # Add statistics text box
    min_loss = min(losses)
    max_loss = max(losses)
    final_loss = losses[-1]
    min_log_loss = min(log_losses)
    max_log_loss = max(log_losses)
    
    stats_text = (f'Original - Min: {min_loss:.4f}, Max: {max_loss:.4f}, Final: {final_loss:.4f}\n'
                  f'Log Scale - Min: {min_log_loss:.4f}, Max: {max_log_loss:.4f}\n'
                  f'Total Points: {len(losses)}')
    
    # Position the text box below the legend in the upper right corner
    ax1.text(0.98, 0.8, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Adjust layout to accommodate the third y-axis
    plt.subplots_adjust(right=0.85)
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved plot: {output_path}")
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
        plt.close()
        
def process_log_files():
    """
    Main function to process all log files and generate loss curves
    """
    # Define input and output directories
    log_dir = Path("logs_processed")
    output_dir = Path("loss_curves")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Check if log directory exists
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist!")
        return
    
    # Find all txt files in log directory
    txt_files = list(log_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in '{log_dir}' directory!")
        return
    
    print(f"Found {len(txt_files)} .txt files to process...")
    
    # Process each file
    for txt_file in txt_files:
        print(f"\nProcessing: {txt_file.name}")
        
        # Parse the log file
        iterations, losses = parse_log_file(txt_file)
        
        if not iterations:
            print(f"No valid log entries found in {txt_file.name}")
            continue
        
        # Create output filename (same name but .png extension)
        output_filename = txt_file.stem + ".png"
        output_path = output_dir / output_filename
        
        # Create and save the plot
        create_loss_curve(iterations, losses, output_path, txt_file.stem)
    
    print(f"\nProcessing complete! Check the '{output_dir}' directory for generated plots.")

if __name__ == "__main__":
    process_log_files()