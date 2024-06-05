#!/bin/bash

directory="logs/*"

fswatch -o "$directory" | while read file
do
    echo "File changed: $file"
done
