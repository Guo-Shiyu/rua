#! /usr/bin/bash 

if [ -z "$1" ]; then 
    echo "Please give new version number, string like \"0.1.23\". "
    exit 1 
fi 

subdirs=$(find . -mindepth  1 -maxdepth 1 -type d)

for dir in $subdirs; do
    if [ -f "$dir/Cargo.toml" ]; then 
        sed -i -E "s/^version = \".*\"$/version = \"$1\"/" "$dir/Cargo.toml"
        echo "set version of $dir/Cargo.toml to $1"
    else 
        echo "skip $dir"
    fi 
done 