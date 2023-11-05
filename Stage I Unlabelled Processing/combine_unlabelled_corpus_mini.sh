#!/bin/bash

# Initialize the output file
output_file="all_text_mini"

# Loop through file names from 0001.txt to 0050.txt
for i in {1..50}; do
    file_number=$(printf "%04d" "$i")  # Zero-pad the file number
    cat "$file_number.txt" >> "$output_file"
done
