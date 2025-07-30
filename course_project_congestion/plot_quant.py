import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime

def parse_quantization_log(file_path):
    """
    Parse quantization log file to extract model test results using line-by-line approach
    """
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_model = None
        current_metrics = {}
        current_time = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check for model testing start
            model_match = re.search(r'--- Testing Quantized Model: (.+?) ---', line)
            if model_match:
                # Save previous model if complete
                if current_model and len(current_metrics) == 3:
                    results.append({
                        'Model_Name': current_model,
                        'Test_Time': current_time or "Unknown",
                        'NRMS': current_metrics['NRMS'],
                        'SSIM': current_metrics['SSIM'],
                        'EMD': current_metrics['EMD']
                    })
                
                # Start new model
                current_model = model_match.group(1).strip()
                current_metrics = {}
                current_time = None
                print(f"  Found model: {current_model}")
                continue
            
            # Skip if no current model
            if not current_model:
                continue
            
            # Check for NRMS
            nrms_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) -   NRMS: ([\d.]+)', line)
            if nrms_match:
                current_time = nrms_match.group(1)
                current_metrics['NRMS'] = float(nrms_match.group(2))
                print(f"    NRMS: {current_metrics['NRMS']}")
                continue
            
            # Check for SSIM
            ssim_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -   SSIM: ([\d.]+)', line)
            if ssim_match:
                current_metrics['SSIM'] = float(ssim_match.group(1))
                print(f"    SSIM: {current_metrics['SSIM']}")
                continue
            
            # Check for EMD
            emd_match = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} -   EMD: ([\d.]+)', line)
            if emd_match:
                current_metrics['EMD'] = float(emd_match.group(1))
                print(f"    EMD: {current_metrics['EMD']}")
                continue
        
        # Don't forget the last model
        if current_model and len(current_metrics) == 3:
            results.append({
                'Model_Name': current_model,
                'Test_Time': current_time or "Unknown",
                'NRMS': current_metrics['NRMS'],
                'SSIM': current_metrics['SSIM'],
                'EMD': current_metrics['EMD']
            })
        
        print(f"  Successfully parsed {len(results)} complete results")
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    return results

def create_individual_heatmaps(df, output_dir):
    """
    Create individual heatmaps for each metric with INT4 and INT8 in separate columns
    """
    metrics = ['NRMS', 'SSIM', 'EMD']
    colors = ['Reds_r', 'Greens', 'Blues_r']  # Reverse for NRMS and EMD (lower is better)
    
    # Sort data by model name to ensure consistent ordering
    model_order = sorted(df['Model_Name'].unique())
    quant_order = ['INT4', 'INT8']
    
    for metric, cmap in zip(metrics, colors):
        plt.figure(figsize=(8, 12))  # Taller figure for better model name visibility
        
        # Create pivot table for this metric with ordered data
        pivot_data = df.pivot(index='Model_Name', columns='Quantization_Type', values=metric)
        pivot_data = pivot_data.reindex(index=model_order, columns=quant_order)
        
        # Create heatmap with tight spacing
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap=cmap, 
                   cbar_kws={'label': f'{metric} Value', 'shrink': 0.8}, 
                   linewidths=0.5, linecolor='white',
                   square=False)
        
        # Customize the plot
        better_text = "Lower is Better" if metric in ['NRMS', 'EMD'] else "Higher is Better"
        plt.title(f'{metric} Performance Heatmap\n({better_text})', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Quantization Type', fontsize=14)
        plt.ylabel('Model Name', fontsize=14)
        
        # Rotate labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save individual heatmap
        heatmap_path = output_dir / f"heatmap_{metric.lower()}.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{metric} heatmap saved: {heatmap_path}")

def generate_markdown_report(all_results, output_path, output_dir):
    """
    Generate comprehensive markdown report with individual heatmaps only (no test time)
    """
    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No quantization results found to generate report!")
        return
    
    # Sort by quantization type and model name
    df = df.sort_values(['Model_Name', 'Quantization_Type'])
    
    # Start writing markdown content
    md_content = []
    
    # Header
    md_content.append("# Quantization Results Summary")
    md_content.append(f"\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append(f"\n**Total Models Tested:** {len(df)}")
    md_content.append(f"\n**Quantization Types:** {', '.join(sorted(df['Quantization_Type'].unique()))}")
    md_content.append("\n---\n")
    
    # Add individual metric heatmaps only
    md_content.append("## Visual Performance Comparison")
    md_content.append("\n### Individual Metric Heatmaps")
    md_content.append("#### NRMS Performance")
    md_content.append("![NRMS Heatmap](heatmap_nrms.png)")
    md_content.append("\n#### SSIM Performance")
    md_content.append("![SSIM Heatmap](heatmap_ssim.png)")
    md_content.append("\n#### EMD Performance")
    md_content.append("![EMD Heatmap](heatmap_emd.png)")
    md_content.append("\n---\n")
    
    # Table of Contents
    md_content.append("## Table of Contents")
    md_content.append("1. [Visual Performance Comparison](#visual-performance-comparison)")
    md_content.append("2. [Overall Summary](#overall-summary)")
    md_content.append("3. [Performance Analysis](#performance-analysis)")
    md_content.append("4. [Best Performing Models](#best-performing-models)")
    md_content.append("5. [Detailed Results by Quantization Type](#detailed-results-by-quantization-type)")
    md_content.append("\n---\n")
    
    # Overall Summary Table (without test time)
    md_content.append("## Overall Summary")
    md_content.append("\n| Quantization Type | Model Name | NRMS | SSIM | EMD |")
    md_content.append("|-------------------|------------|------|------|-----|")
    
    for _, row in df.iterrows():
        md_content.append(f"| {row['Quantization_Type']} | {row['Model_Name']} | "
                         f"{row['NRMS']:.4f} | {row['SSIM']:.4f} | {row['EMD']:.4f} |")
    
    md_content.append("\n---\n")
    
    # Performance Analysis
    md_content.append("## Performance Analysis")
    
    # Average performance by quantization type
    avg_by_type = df.groupby('Quantization_Type')[['NRMS', 'SSIM', 'EMD']].mean()
    
    md_content.append("\n### Average Performance by Quantization Type")
    md_content.append("\n| Quantization Type | Avg NRMS | Avg SSIM | Avg EMD |")
    md_content.append("|-------------------|----------|----------|---------|")
    
    for quant_type, row in avg_by_type.iterrows():
        md_content.append(f"| {quant_type} | {row['NRMS']:.4f} | {row['SSIM']:.4f} | {row['EMD']:.4f} |")
    
    # Performance metrics explanation
    md_content.append("\n### Metrics Explanation")
    md_content.append("- **NRMS (Normalized Root Mean Square):** Lower values indicate better performance")
    md_content.append("- **SSIM (Structural Similarity Index):** Higher values indicate better performance (range: 0-1)")
    md_content.append("- **EMD (Earth Mover's Distance):** Lower values indicate better performance")
    
    md_content.append("\n---\n")
    
    # Best Performing Models
    md_content.append("## Best Performing Models")
    
    best_nrms = df.loc[df['NRMS'].idxmin()]
    best_ssim = df.loc[df['SSIM'].idxmax()]
    best_emd = df.loc[df['EMD'].idxmin()]
    
    md_content.append(f"\n### 🏆 Best Models by Metric")
    md_content.append(f"\n- **Best NRMS (Lowest):** {best_nrms['Model_Name']} ({best_nrms['Quantization_Type']}) - {best_nrms['NRMS']:.4f}")
    md_content.append(f"- **Best SSIM (Highest):** {best_ssim['Model_Name']} ({best_ssim['Quantization_Type']}) - {best_ssim['SSIM']:.4f}")
    md_content.append(f"- **Best EMD (Lowest):** {best_emd['Model_Name']} ({best_emd['Quantization_Type']}) - {best_emd['EMD']:.4f}")
    
    # Overall best model (composite score)
    df_norm = df.copy()
    df_norm['NRMS_norm'] = (df['NRMS'] - df['NRMS'].min()) / (df['NRMS'].max() - df['NRMS'].min())
    df_norm['SSIM_norm'] = (df['SSIM'] - df['SSIM'].min()) / (df['SSIM'].max() - df['SSIM'].min())
    df_norm['EMD_norm'] = (df['EMD'] - df['EMD'].min()) / (df['EMD'].max() - df['EMD'].min())
    df_norm['composite_score'] = (df_norm['NRMS_norm'] + (1 - df_norm['SSIM_norm']) + df_norm['EMD_norm']) / 3
    best_overall = df_norm.loc[df_norm['composite_score'].idxmin()]
    
    md_content.append(f"\n### 🎯 Overall Best Model")
    md_content.append(f"**{best_overall['Model_Name']}** ({best_overall['Quantization_Type']})")
    md_content.append(f"- NRMS: {best_overall['NRMS']:.4f}")
    md_content.append(f"- SSIM: {best_overall['SSIM']:.4f}")
    md_content.append(f"- EMD: {best_overall['EMD']:.4f}")
    md_content.append(f"- Composite Score: {best_overall['composite_score']:.4f}")
    
    md_content.append("\n---\n")
    
    # Detailed Results by Quantization Type
    md_content.append("## Detailed Results by Quantization Type")
    
    for quant_type in sorted(df['Quantization_Type'].unique()):
        type_df = df[df['Quantization_Type'] == quant_type].sort_values('Model_Name')
        
        md_content.append(f"\n### {quant_type}")
        md_content.append(f"\n**Number of models:** {len(type_df)}")
        
        # Detailed table for this type (without test time)
        md_content.append(f"\n#### Results")
        md_content.append("\n| Model Name | NRMS | SSIM | EMD |")
        md_content.append("|------------|------|------|-----|")
        
        for _, row in type_df.iterrows():
            md_content.append(f"| {row['Model_Name']} | {row['NRMS']:.4f} | "
                             f"{row['SSIM']:.4f} | {row['EMD']:.4f} |")
        
        md_content.append("")
    
    # Footer
    md_content.append(f"\n---\n")
    md_content.append("*Report generated automatically from quantization log files*")
    
    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_content))
        print(f"Successfully generated markdown report: {output_path}")
    except Exception as e:
        print(f"Error writing markdown file: {e}")

def process_quantization_logs():
    """
    Main function to process all quantization log files and generate markdown report with plots
    """
    # Define input and output directories
    log_dir = Path("logs")
    output_dir = Path("quantization_results")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Check if log directory exists
    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist!")
        return
    
    # Find all quantization log files
    quant_files = list(log_dir.glob("quant_log_*.txt"))
    
    if not quant_files:
        print(f"No quantization log files found in '{log_dir}' directory!")
        return
    
    print(f"Found {len(quant_files)} quantization log files to process...")
    
    all_results = []
    
    # Process each file
    for log_file in quant_files:
        print(f"\nProcessing: {log_file.name}")
        
        # Extract quantization type from filename
        quant_type = log_file.stem.replace('quant_log_', '')
        
        # Parse the log file
        results = parse_quantization_log(log_file)
        
        if not results:
            print(f"No valid quantization results found in {log_file.name}")
            continue
        
        # Add quantization type to each result
        for result in results:
            result['Quantization_Type'] = quant_type
            all_results.append(result)
        
        print(f"  Found {len(results)} quantized models")
    
    if not all_results:
        print("No quantization results found!")
        return
    
    # Create DataFrame for plotting
    df = pd.DataFrame(all_results)
    
    # Generate individual heatmaps only
    print("\nGenerating individual heatmaps...")
    create_individual_heatmaps(df, output_dir)
    
    # Generate markdown report
    output_path = output_dir / "quantization_summary.md"
    generate_markdown_report(all_results, output_path, output_dir)
    
    print(f"\nProcessing complete! Check the '{output_dir}' directory for generated report and plots.")

if __name__ == "__main__":
    process_quantization_logs()