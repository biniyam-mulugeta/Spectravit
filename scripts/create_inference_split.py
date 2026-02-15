import os
import shutil
import random
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create an inference dataset by extracting a subset of data.")
    parser.add_argument("--source", required=True, help="Path to the source dataset")
    parser.add_argument("--dest", required=True, help="Path to the destination inference dataset")
    parser.add_argument("--percentage", type=float, default=20.0, help="Percentage of data to extract (0-100)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (removes from source)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    source_path = Path(args.source)
    dest_path = Path(args.dest)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_path}' does not exist.")
        return

    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    print(f"Operation: {'Moving' if args.move else 'Copying'} {args.percentage}% of files")
    
    count = 0
    processed = 0
    
    # Walk the tree to preserve structure
    for root, dirs, files in os.walk(source_path):
        # Filter out hidden files
        files = [f for f in files if not f.startswith('.')]
        
        if not files:
            continue
            
        # Determine how many files to take from this folder
        n_files = len(files)
        n_extract = int(n_files * (args.percentage / 100.0))
        
        if n_extract > 0:
            # Randomly select files
            selected_files = random.sample(files, n_extract)
            
            # Calculate destination folder
            rel_path = Path(root).relative_to(source_path)
            target_dir = dest_path / rel_path
            
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)
            
            for file_name in selected_files:
                src_file = Path(root) / file_name
                dst_file = target_dir / file_name
                
                if args.move:
                    shutil.move(str(src_file), str(dst_file))
                else:
                    shutil.copy2(str(src_file), str(dst_file))
                
                processed += 1
        
        count += n_files

    print(f"Processing complete.")
    print(f"Total files scanned: {count}")
    print(f"Files {'moved' if args.move else 'copied'} to inference set: {processed}")

if __name__ == "__main__":
    main()