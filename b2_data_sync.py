#!/usr/bin/env python3
"""
Backblaze B2 sync script for managing data files and model checkpoints across devices.

Usage:
    python b2_data_sync.py download                        # Download all files from B2 to data/
    python b2_data_sync.py upload <file1> <file2> ...     # Upload specific files to B2
    python b2_data_sync.py upload data/*                   # Upload all files in data/
    python b2_data_sync.py list                            # List files in the bucket
    python b2_data_sync.py upload-model <dir_uuid>        # Upload model directory to B2
    python b2_data_sync.py get-model <dir_uuid>           # Download model directory from B2
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from b2sdk.v2 import B2Api, InMemoryAccountInfo
from dotenv import load_dotenv
from tqdm import tqdm


class B2Sync:
    """Handle Backblaze B2 bucket synchronization."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        load_dotenv(dotenv_path=self.project_root / ".env")

        self.app_key_id = os.getenv("B2_APP_KEY_ID")
        self.app_key = os.getenv("B2_APP_KEY")
        self.bucket_name = os.getenv("B2_BUCKET_NAME")

        if not all([self.app_key_id, self.app_key, self.bucket_name]):
            print("Error: Missing B2 credentials in .env file")
            print("Required variables: B2_APP_KEY_ID, B2_APP_KEY, B2_BUCKET_NAME")
            sys.exit(1)

        # Initialize B2 API
        info = InMemoryAccountInfo()
        self.api = B2Api(info)
        self.api.authorize_account("production", self.app_key_id, self.app_key)
        self.bucket = self.api.get_bucket_by_name(self.bucket_name)

    def upload(self, files: list[str]):
        """Upload specific files to B2."""
        if not files:
            print("Error: No files specified for upload")
            print("Usage: python b2_data_sync.py upload <file1> <file2> ...")
            sys.exit(1)

        # Expand glob patterns and resolve paths
        file_paths = []
        for file_pattern in files:
            # Use glob.glob for robust pattern matching
            matches = glob.glob(file_pattern, recursive=True)
            if matches:
                file_paths.extend([Path(f) for f in matches if Path(f).is_file()])
            else:
                # No matches - check if it's a literal file path
                path = Path(file_pattern)
                if path.exists() and path.is_file():
                    file_paths.append(path)
                else:
                    print(f"Warning: File not found or is not a file: {file_pattern}")

        if not file_paths:
            print("No valid files found to upload")
            return

        print(f"Uploading {len(file_paths)} file(s)...")

        for file_path in tqdm(file_paths, desc="Uploading"):
            # Upload with just the filename (no directory path)
            b2_file_name = file_path.name

            # Upload file
            try:
                self.bucket.upload_local_file(local_file=str(file_path), file_name=b2_file_name)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                tqdm.write(f"Uploaded: {b2_file_name} ({size_mb:.2f} MB)")
            except Exception as e:
                tqdm.write(f"Failed to upload {file_path}: {e}")

        print("\nUpload complete")

    def download(self):
        """Download all files from B2 to local data/ directory."""
        print("Fetching file list from B2...")

        # List all files in bucket
        files_in_bucket = list(self.bucket.ls())

        if not files_in_bucket:
            print("No files found in bucket")
            return

        print(f"Found {len(files_in_bucket)} file(s) in bucket")

        # Create data directory if it doesn't exist
        data_dir = self.project_root / "data"
        data_dir.mkdir(exist_ok=True)

        for file_version_info, _ in tqdm(files_in_bucket, desc="Downloading"):
            b2_file_name = file_version_info.file_name
            # Download to data/ directory
            local_path = data_dir / b2_file_name

            # Validate path to prevent directory traversal attacks
            try:
                resolved_path = local_path.resolve()
                data_dir_resolved = data_dir.resolve()
                if not resolved_path.is_relative_to(data_dir_resolved):
                    tqdm.write(f"Skipping {b2_file_name}: path traversal attempt detected")
                    continue
            except (ValueError, OSError) as e:
                tqdm.write(f"Skipping {b2_file_name}: invalid path ({e})")
                continue

            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            try:
                downloaded_file = self.bucket.download_file_by_name(b2_file_name)
                downloaded_file.save_to(str(local_path))
                size_mb = file_version_info.size / (1024 * 1024)
                tqdm.write(f"Downloaded: {b2_file_name} ({size_mb:.2f} MB)")
            except Exception as e:
                tqdm.write(f"Failed to download {b2_file_name}: {e}")

        print("\nDownload complete")

    def list_files(self):
        """List all files in the B2 bucket."""
        print(f"Files in bucket '{self.bucket_name}':\n")

        total_size = 0
        file_count = 0

        for file_version_info, _ in self.bucket.ls():
            size_mb = file_version_info.size / (1024 * 1024)
            total_size += file_version_info.size
            file_count += 1
            print(f"  {file_version_info.file_name:<50} {size_mb:>10.2f} MB")

        if file_count == 0:
            print("  (empty)")
        else:
            total_gb = total_size / (1024 * 1024 * 1024)
            print(f"\nTotal: {file_count} file(s), {total_gb:.2f} GB")

    def upload_model(self, dir_uuid: str):
        """Upload a model directory to B2."""
        models_dir = self.project_root / "models"
        model_dir = models_dir / dir_uuid

        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            sys.exit(1)

        if not model_dir.is_dir():
            print(f"Error: Not a directory: {model_dir}")
            sys.exit(1)

        # Get all files in the model directory
        files = list(model_dir.glob("*"))
        files = [f for f in files if f.is_file()]

        if not files:
            print(f"No files found in {model_dir}")
            return

        print(f"Uploading model directory '{dir_uuid}' ({len(files)} file(s))...")

        for file_path in tqdm(files, desc="Uploading"):
            # Upload with models/dir_uuid/ prefix to preserve directory structure
            b2_file_name = f"models/{dir_uuid}/{file_path.name}"

            try:
                self.bucket.upload_local_file(local_file=str(file_path), file_name=b2_file_name)
                size_mb = file_path.stat().st_size / (1024 * 1024)
                tqdm.write(f"Uploaded: {b2_file_name} ({size_mb:.2f} MB)")
            except Exception as e:
                tqdm.write(f"Failed to upload {file_path}: {e}")

        print("\nModel upload complete")

    def get_model(self, dir_uuid: str):
        """Download a model directory from B2."""
        print(f"Fetching model directory '{dir_uuid}' from B2...")

        # List all files with the models/dir_uuid/ prefix
        prefix = f"models/{dir_uuid}/"
        files_in_bucket = []

        for file_version_info, _ in self.bucket.ls():
            if file_version_info.file_name.startswith(prefix):
                files_in_bucket.append(file_version_info)

        if not files_in_bucket:
            print(f"No files found in B2 with prefix '{prefix}'")
            return

        print(f"Found {len(files_in_bucket)} file(s) for model '{dir_uuid}'")

        # Create model directory if it doesn't exist
        models_dir = self.project_root / "models"
        model_dir = models_dir / dir_uuid
        model_dir.mkdir(parents=True, exist_ok=True)

        for file_version_info in tqdm(files_in_bucket, desc="Downloading"):
            b2_file_name = file_version_info.file_name
            # Extract just the filename (remove models/dir_uuid/ prefix)
            filename = b2_file_name.replace(prefix, "")
            local_path = model_dir / filename

            # Validate path to prevent directory traversal attacks
            try:
                resolved_path = local_path.resolve()
                model_dir_resolved = model_dir.resolve()
                if not resolved_path.is_relative_to(model_dir_resolved):
                    tqdm.write(f"Skipping {b2_file_name}: path traversal attempt detected")
                    continue
            except (ValueError, OSError) as e:
                tqdm.write(f"Skipping {b2_file_name}: invalid path ({e})")
                continue

            # Download file
            try:
                downloaded_file = self.bucket.download_file_by_name(b2_file_name)
                downloaded_file.save_to(str(local_path))
                size_mb = file_version_info.size / (1024 * 1024)
                tqdm.write(f"Downloaded: {filename} ({size_mb:.2f} MB)")
            except Exception as e:
                tqdm.write(f"Failed to download {b2_file_name}: {e}")

        print("\nModel download complete")


def main():
    parser = argparse.ArgumentParser(
        description="Sync data files and model checkpoints with Backblaze B2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python b2_data_sync.py download                               # Download all files from B2 to data/
  python b2_data_sync.py upload data/owt_train.txt              # Upload a specific file
  python b2_data_sync.py upload data/*.npy                      # Upload all .npy files in data/
  python b2_data_sync.py upload *.pickle data/*.txt             # Upload multiple patterns
  python b2_data_sync.py list                                   # List files in bucket
  python b2_data_sync.py upload-model 9d388728-c441-4e93-9d61   # Upload model directory to B2
  python b2_data_sync.py get-model 9d388728-c441-4e93-9d61      # Download model directory from B2

Setup:
  1. Install dependencies: uv sync
  2. Create .env file in project root with:
       B2_APP_KEY_ID=your_key_id
       B2_APP_KEY=your_application_key
       B2_BUCKET_NAME=your_bucket_name
        """,
    )

    parser.add_argument(
        "command",
        choices=["download", "upload", "list", "upload-model", "get-model"],
        help="Command to execute",
    )

    parser.add_argument(
        "files",
        nargs="*",
        help="Files to upload (for upload), or directory UUID (for upload-model/get-model)",
    )

    args = parser.parse_args()

    # Validate that download/list commands don't have extra arguments
    if args.command in ["download", "list"] and args.files:
        parser.error(f"{args.command} command does not accept file arguments")

    # Validate that upload-model/get-model have exactly one argument
    if args.command in ["upload-model", "get-model"]:
        if len(args.files) != 1:
            parser.error(f"{args.command} command requires exactly one directory UUID argument")

    try:
        sync = B2Sync()

        if args.command == "download":
            sync.download()
        elif args.command == "upload":
            sync.upload(args.files)
        elif args.command == "list":
            sync.list_files()
        elif args.command == "upload-model":
            sync.upload_model(args.files[0])
        elif args.command == "get-model":
            sync.get_model(args.files[0])

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
