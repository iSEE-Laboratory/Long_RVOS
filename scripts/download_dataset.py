#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download, login
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_tar(tar_path, extract_to, remove_tar=False):
    """Extract tar.gz files"""
    try:
        logger.info(f"Extracting: {tar_path} -> {extract_to}")
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Show progress
            members = tar.getmembers()
            for member in tqdm(members, desc=f"Extracting {tar_path.name}", unit="files"):
                tar.extract(member, extract_to)
        logger.info(f"✓ Extraction completed: {tar_path.name}")
        
        if remove_tar:
            tar_path.unlink()
            logger.info(f"✓ Deleted archive file: {tar_path.name}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Extraction failed {tar_path}: {e}")
        return False


def download_and_extract_split(repo_id, split, output_dir, remove_tar=False):
    """Download and extract data for specified split"""
    split_output_dir = Path(output_dir) / "long_rvos" / split
    split_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary download directory
    temp_download_dir = Path(output_dir) / ".hf_downloads"
    temp_download_dir.mkdir(parents=True, exist_ok=True)
    
    files = ["JPEGImages.tar.gz", "Annotations.tar.gz"]
    success_count = 0
    
    for filename in files:
        try:
            # Download file
            logger.info(f"Downloading: {split}/{filename}")
            tar_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{split}/{filename}",
                repo_type="dataset",
                local_dir=str(temp_download_dir),
                local_dir_use_symlinks=False
            )
            
            tar_path = Path(tar_path)
            
            # Verify file exists
            if not tar_path.exists():
                logger.error(f"Downloaded file does not exist: {tar_path}")
                continue
            
            # Extract to correct location
            if "JPEGImages" in filename:
                extract_to = split_output_dir / "JPEGImages"
            elif "Annotations" in filename:
                extract_to = split_output_dir / "Annotations"
            else:
                extract_to = split_output_dir
            
            extract_to.mkdir(parents=True, exist_ok=True)
            
            # Extract
            if extract_tar(tar_path, extract_to, remove_tar=remove_tar):
                success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Download/extraction failed {split}/{filename}: {e}")
    
    # Clean up temporary directory (if empty)
    try:
        if temp_download_dir.exists() and not any(temp_download_dir.iterdir()):
            temp_download_dir.rmdir()
    except Exception:
        pass
    
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Download and extract dataset from Hugging Face Hub")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Hugging Face repository ID, format: username/dataset-name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory path (default: data)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to download (default: train valid test)"
    )
    parser.add_argument(
        "--remove_tar",
        action="store_true",
        help="Delete archive files after extraction to save space"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (only needed for private datasets, public datasets don't require authentication)"
    )
    parser.add_argument(
        "--use_snapshot",
        action="store_true",
        help="Use snapshot_download to download entire repository (faster but requires more space)"
    )
    
    args = parser.parse_args()
    
    # Login only if token is provided (not needed for public datasets)
    if args.token:
        login(token=args.token)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.use_snapshot:
        # Use snapshot_download to download entire repository
        logger.info("Using snapshot_download to download entire repository...")
        try:
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                local_dir=output_dir / "hf_dataset",
                local_dir_use_symlinks=False
            )
            logger.info("✓ Download completed!")
            
            # Extract all files
            hf_dataset_dir = output_dir / "hf_dataset"
            for split in args.splits:
                split_dir = hf_dataset_dir / split
                if not split_dir.exists():
                    continue
                
                split_output_dir = output_dir / "long_rvos" / split
                split_output_dir.mkdir(parents=True, exist_ok=True)
                
                for tar_file in split_dir.glob("*.tar.gz"):
                    if "JPEGImages" in tar_file.name:
                        extract_to = split_output_dir / "JPEGImages"
                    elif "Annotations" in tar_file.name:
                        extract_to = split_output_dir / "Annotations"
                    else:
                        extract_to = split_output_dir
                    
                    extract_to.mkdir(parents=True, exist_ok=True)
                    extract_tar(tar_file, extract_to, remove_tar=args.remove_tar)
            
            logger.info("✓ Extraction completed!")
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return
    else:
        # Download files one by one
        total_success = 0
        total_files = len(args.splits) * 2  # Each split has 2 files
        
        for split in args.splits:
            logger.info(f"\nProcessing split: {split}")
            success = download_and_extract_split(
                args.repo_id,
                split,
                args.output_dir,
                remove_tar=args.remove_tar
            )
            total_success += success
        
        logger.info(f"\nCompleted! Successfully processed: {total_success}/{total_files} files")
    
    # Validate directory structure
    logger.info("\nValidating directory structure...")
    base_dir = output_dir / "long_rvos"
    
    for split in args.splits:
        split_dir = base_dir / split
        if split_dir.exists():
            required_dirs = ["JPEGImages", "Annotations"]
            for req_dir in required_dirs:
                if (split_dir / req_dir).exists():
                    logger.info(f"✓ {split}/{req_dir} exists")
                else:
                    logger.warning(f"✗ {split}/{req_dir} does not exist")
        else:
            logger.warning(f"✗ {split} directory does not exist")
    
    logger.info("\nDataset preparation completed!")
    logger.info(f"Dataset located at: {base_dir.absolute()}")


if __name__ == "__main__":
    main()

