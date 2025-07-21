import re
import sys

def compute_average_success(file_path):
    # List to store all the extracted numbers
    success_values = []

    # Open the file and process line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Search for 'overall success = {number}'
            match = re.search(r'overall success rate: (\d+(\.\d+)?)', line)
            if match:
                # Extract the number and convert to float
                success_value = float(match.group(1))
                success_values.append(success_value)

    # Compute the average if there are values
    if success_values:
        average_success = sum(success_values) * 100 / len(success_values)
        print(f"Average overall success: {average_success:.2f}")
        print(f"len(success_values): {len(success_values)}")
    else:
        print("No 'overall success = {number}' found in the file.")

# Specify the path to your file
file_path = sys.argv[1]

# Call the function
compute_average_success(file_path)