#!/bin/bash

# Define the directory and output file
directory="./results/stage_one"  # Replace with your folder path
output_file="results/stage_one/all_files_content.txt"

# Create (or overwrite) the output file
: > "$output_file"

# Loop through each file in the directory
for file in "$directory"/*; do
	if [ -f "$file" ]; then
		# Write the filename to the output file
		echo "Filename: $(basename "$file")" >> "$output_file"
		echo "----------------------------------------" >> "$output_file"
			          
		# Write the contents of the file to the output file
		cat "$file" >> "$output_file"
				          
		# Add a separator between files
		echo -e "\n========================================\n" >> "$output_file"
		fi
done

echo "Contents of all files written to $output_file."

