#!/bin/bash

timeseries_length="3"
modes=("train" "val" "test")
wds_dir="webdatasets"  # The directory to save the webdatasets into

for mode in "${modes[@]}"; do
    if [ "$mode" == "train" ]; then
        submodes=("train_pos" "train_neg")
    else
        submodes=("$mode")
    fi

    for submode in "${submodes[@]}"; do
        base_dir="${wds_dir}/${timeseries_length}/${submode}"
        pattern="sample-${submode}"
        base_name="${base_dir}/${pattern}"

        index=0

        for file in ${base_name}-*.*; do
            extension="${file##*.}"
            new_name="${base_name}-$(printf "%06d" $index).${extension}"
            mv "$file" "$new_name"
            ((index++))
        done

        echo "Files renamed for pattern ${pattern} successfully."
    done
done
