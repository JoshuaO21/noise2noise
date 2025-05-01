import numpy as np
import matplotlib.pyplot as plt
import re
import sys

def parse_psnr_file(file_path):
    """
    Parse PSNR values from a file with format:
    index: value1 value2 value3
    
    Returns:
        A dictionary of numpy arrays, one for each column of values
    """
    values = {
        'noisy': [],
        'denoised_clamped': [],
        'denoised_psnr': []
    }
    
    indices = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Extract values using regex
            match = re.match(r'\s*(\d+):\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)', line)
            if match:
                idx, val1, val2, val3 = match.groups()
                indices.append(int(idx))
                values['noisy'].append(float(val1))
                values['denoised_clamped'].append(float(val2))
                values['denoised_psnr'].append(float(val3))
    
    # Convert to numpy arrays
    for key in values:
        values[key] = np.array(values[key])
    
    return values, indices

def calculate_statistics(values):
    """
    Calculate statistics for each column of PSNR values
    """
    stats = {}
    
    for key, data in values.items():
        stats[key] = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'q1': np.percentile(data, 25),
            'q3': np.percentile(data, 75)
        }
    
    return stats

def print_statistics(stats):
    """
    Print statistics in a readable format
    """
    print("\n===== PSNR Statistics =====\n")
    
    for col_name, stat in stats.items():
        print(f"Column: {col_name}")
        print(f"  Mean PSNR:     {stat['mean']:.4f} dB")
        print(f"  Median PSNR:   {stat['median']:.4f} dB")
        print(f"  Std Dev:       {stat['std']:.4f} dB")
        print(f"  Min PSNR:      {stat['min']:.4f} dB")
        print(f"  Max PSNR:      {stat['max']:.4f} dB")
        print(f"  Range:         {stat['range']:.4f} dB")
        print(f"  25th Percentile: {stat['q1']:.4f} dB")
        print(f"  75th Percentile: {stat['q3']:.4f} dB")
        print()

def plot_histograms(values):
    """
    Plot histograms of PSNR values
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    titles = ['Noisy', 'Denoised Method 1', 'Denoised Method 2']
    data_keys = ['noisy', 'denoised_1', 'denoised_2']
    
    for i, (title, key) in enumerate(zip(titles, data_keys)):
        axes[i].hist(values[key], bins=10, alpha=0.7, color=f'C{i}')
        axes[i].axvline(np.mean(values[key]), color='red', linestyle='dashed', linewidth=1)
        axes[i].set_title(f'{title} (Mean: {np.mean(values[key]):.2f} dB)')
        axes[i].set_xlabel('PSNR (dB)')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('psnr_histograms.png')
    print("Saved histogram plot to 'psnr_histograms.png'")

def plot_comparison(values, indices):
    """
    Plot comparison of PSNR values across all test images
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(indices, values['noisy'], 'o-', label='Noisy', alpha=0.7)
    plt.plot(indices, values['denoised_1'], 's-', label='Clamped PSNR', alpha=0.7)
    plt.plot(indices, values['denoised_2'], '^-', label='Denoised PSNR', alpha=0.7)
    
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR Comparison Across Test Images')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('psnr_comparison.png')
    print("Saved comparison plot to 'psnr_comparison.png'")

def main():
    # Get the file path from command line argument or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "psnr.txt"
    
    try:
        # Parse the file
        values, indices = parse_psnr_file(file_path)
        
        # Calculate statistics
        stats = calculate_statistics(values)
        
        # Print statistics
        print_statistics(stats)
        
        # Plot histograms
        plot_histograms(values)
        
        # Plot comparison
        plot_comparison(values, indices)

    except Exception as e:
        print(f"Error processing the file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())