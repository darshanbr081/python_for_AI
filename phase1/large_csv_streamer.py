""" **Large CSV Streamer:** Write a Python generator to process a 10GB CSV file line-by-line to calculate the running mean of a column."""

import csv

def large_csv_streamer(file_path, column_index):
    """
    A generator function that processes a large CSV file line-by-line to calculate the running mean of a specified column.

    :param file_path: Path to the CSV file.
    :param column_index: The index of the column for which to calculate the running mean.
    :yield: The running mean of the specified column after processing each line.
    """
    total = 0.0
    count = 0
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                value = float(row[column_index])
                total += value
                count += 1
                running_mean = total / count
                yield running_mean
            except (ValueError, IndexError):
                # skip lines that cannot be processed
            continue

# Example usage:
#for mean in large_csv_streamer('large_file.csv', 2):
#print(mean)

