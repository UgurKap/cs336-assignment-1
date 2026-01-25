#!/usr/bin/env python3
"""
Backblaze B2 sync script for managing data files across devices.

Usage:
    python b2_data_sync.py download                        # Download all files from B2 to data/
    python b2_data_sync.py upload <file1> <file2> ...     # Upload specific files to B2
    python b2_data_sync.py upload data/*                   # Upload all files in data/
    python b2_data_sync.py list                            # List files in the bucket
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


def main():
    parser = argparse.ArgumentParser(
        description="Sync data files with Backblaze B2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python b2_data_sync.py download                     # Download all files from B2 to data/
  python b2_data_sync.py upload data/owt_train.txt    # Upload a specific file
  python b2_data_sync.py upload data/*.npy            # Upload all .npy files in data/
  python b2_data_sync.py upload *.pickle data/*.txt   # Upload multiple patterns
  python b2_data_sync.py list                         # List files in bucket

Setup:
  1. Install dependencies: uv sync
  2. Create .env file in project root with:
       B2_APP_KEY_ID=your_key_id
       B2_APP_KEY=your_application_key
       B2_BUCKET_NAME=your_bucket_name
        """,
    )

    parser.add_argument("command", choices=["download", "upload", "list"], help="Command to execute")

    parser.add_argument(
        "files", nargs="*", help="Files to upload (required for upload command, supports glob patterns)"
    )

    args = parser.parse_args()

    # Validate that download/list commands don't have extra arguments
    if args.command in ["download", "list"] and args.files:
        parser.error(f"{args.command} command does not accept file arguments")

    try:
        sync = B2Sync()

        if args.command == "download":
            sync.download()
        elif args.command == "upload":
            sync.upload(args.files)
        elif args.command == "list":
            sync.list_files()

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
