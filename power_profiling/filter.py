import os
import pandas as pd
import argparse
import glob


def filter_power_spikes(series, alpha=0.5):
    filtered_values = series.copy()
    for i in range(1, len(series)):
        filtered_values[i] = alpha * filtered_values[i - 1] + (1 - alpha) * series[i]
    return filtered_values


def find_start_index(df):
    for i in range(1, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        if (prev_row['SQ_BUSY_CYCLES'] == 0 and curr_row['SQ_BUSY_CYCLES'] > 0):
            return i
    return 0


def process_file(data_path):
    try:
        # Read CSV data
        df = pd.read_csv(data_path)
        base_name = os.path.basename(data_path)
        print(f"Processing {base_name}...")

        # Find start index and trim data
        start = find_start_index(df)
        df = df.iloc[start:].reset_index(drop=True)

        # Get the start timestamp
        start_timestamp = df.iloc[0]['timestamp'] if 'timestamp' in df.columns else 0

        # Calculate relative timestamps (time since start)
        if 'timestamp' in df.columns:
            df['relative_timestamp'] = df['timestamp'] - start_timestamp
        else:
            df['relative_timestamp'] = df.index  # Use index if no timestamp column exists

        # Process and filter data
        power_data = filter_power_spikes(df['power_from_e'])

        # Create output dataframe with filtered power and timestamp
        output_df = pd.DataFrame({
            'power_from_e': power_data,
            'timestamp': df['relative_timestamp']
        })

        # Save filtered data
        output_file = base_name + '_filtered.csv'
        output_df.to_csv(output_file, index=False)
        print(f"Filtered data saved to: {output_file}")
        
        return True
    except Exception as e:
        print(f"Error processing {data_path}: {str(e)}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process CSV files with power data')
    parser.add_argument('--dir', default='.', help='Directory to search for files (default: current directory)')
    parser.add_argument('--pattern', default='*.csv', help='File pattern to match (default: *.csv)')
    
    args = parser.parse_args()
    
    # Change to the specified directory
    if args.dir != '.':
        os.chdir(args.dir)
    
    # Find files matching the pattern
    file_pattern = args.pattern
    data_paths = glob.glob(file_pattern)
    
    if not data_paths:
        print(f"No files found matching pattern '{file_pattern}' in directory '{args.dir}'")
        return
    
    print(f"Found {len(data_paths)} files matching pattern '{file_pattern}'")
    print(f"Working directory: {os.getcwd()}")
    print("Processing files...")
    
    processed = 0
    for path in data_paths:
        if process_file(path):
            processed += 1
    
    print(f"\n{processed} out of {len(data_paths)} files processed successfully!")


if __name__ == "__main__":
    main()