#!/usr/bin/env python3
import os
import shutil
import argparse
import glob

def clean_build_dir(build_dir, dry_run=False):
    """
    Surgically cleans a Vivado/Vitis build directory to save space.
    Keeps: .xclbin, .xclbin.info, .ltx, and all .rpt/.log files.
    """
    if not os.path.exists(build_dir):
        print(f"Error: Directory {build_dir} does not exist.")
        return

    keep_extensions = {'.xclbin', '.info', '.ltx', '.rpt', '.log', '.pb', '.pb3', '.json'}
    
    files_to_keep = []
    for root, dirs, files in os.walk(build_dir):
        for file in files:
            ext = os.path.splitext(file)[1]
            # Handle .xclbin.info specifically if needed, but .info covers it
            if ext in keep_extensions or file.endswith('.xclbin.info'):
                files_to_keep.append(os.path.join(root, file))

    print(f"Found {len(files_to_keep)} files to keep in {build_dir}")
    
    # We will delete everything in build_dir EXCEPT the files in files_to_keep
    # and the directories containing them.
    
    all_items = []
    for root, dirs, files in os.walk(build_dir, topdown=False):
        for name in files:
            all_items.append(os.path.join(root, name))
        for name in dirs:
            all_items.append(os.path.join(root, name))

    keep_set = set(files_to_keep)
    # Also keep parent directories of kept files
    for f in files_to_keep:
        parent = os.path.dirname(f)
        while parent.startswith(build_dir):
            keep_set.add(parent)
            parent = os.path.dirname(parent)

    deleted_count = 0
    deleted_size = 0
    
    for item in all_items:
        if item not in keep_set:
            try:
                size = 0
                if os.path.isfile(item) or os.path.islink(item):
                    size = os.path.getsize(item)
                    if not dry_run:
                        os.remove(item)
                elif os.path.isdir(item):
                    # Only delete if it's not in keep_set (which includes parents of kept files)
                    if not dry_run:
                        shutil.rmtree(item)
                
                deleted_count += 1
                deleted_size += size
            except Exception as e:
                if not dry_run:
                    print(f"Failed to delete {item}: {e}")

    mode = "Dry-run: Would have deleted" if dry_run else "Deleted"
    print(f"{mode} {deleted_count} items, saving approx {deleted_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Vivado build directory.")
    parser.add_argument("build_dir", help="Path to the build directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    args = parser.parse_args()
    
    clean_build_dir(args.build_dir, args.dry_run)
