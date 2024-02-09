#!/bin/bash

# Infinite loop to continuously install .whl files
while true; do
    # Loop through each .whl file in the directory and install it
    for file in *.whl; do
        echo "Installing $file..."
        pip install "$file"
    done

    sleep 30 # Wait for 60 seconds before checking again
done

