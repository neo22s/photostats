#!/usr/bin/env python3
import os
import sys
import time
import json
import sqlite3
import subprocess
import logging
import multiprocessing
from multiprocessing import Manager, Process, Pool
from collections import Counter
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# ------------------------------------------------------------------------------
# Constants and Global Configuration
# ------------------------------------------------------------------------------
ERROR_LOG = "photo_stats_errors.log"
DB_FILE = "photo_stats_cache.db"
BATCH_SIZE = 50  # Number of files to process per exiftool batch call
DB_WRITE_BATCH_SIZE = 100  # Number of records to insert before a commit

# Allowed file extensions for RAW and JPEG files
RAW_EXTENSIONS = {"cr2", "cr3", "nef", "arw", "raf", "dng", "rw2"}
JPEG_EXTENSIONS = {"jpg", "jpeg"}
ALLOWED_EXTENSIONS = RAW_EXTENSIONS.union(JPEG_EXTENSIONS)

# Credit message to display at start and end of the program
CREDITS = (
    "Developed by @chema_photo - Follow me on Instagram and YouTube.\n"
    "More info about the script at [chemaPhoto](https://chemaphoto.com)\n"
)

# Configure logging for errors with a standard format
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Mapping for white balance normalization
WB_MAPPING: Dict[str, str] = {
    "auto": "auto",
    "auto (ambience priority)": "auto",
    "daylight": "daylight",
    "cloudy": "cloudy",
    "fluorescent": "fluorescent",
    "tungsten": "tungsten",
    "shade": "shade",
    "manual": "manual",
    "manual temperature (kelvin)": "manual"
}

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable time format.
    For seconds < 60: display as seconds with one decimal,
    for minutes and hours use appropriate units.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"

# ------------------------------------------------------------------------------
# Database Functions
# ------------------------------------------------------------------------------
def create_tables_if_needed() -> None:
    """
    Create the necessary database tables if they do not already exist.
    This function uses a context manager to open the database connection.
    """
    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                source_file TEXT PRIMARY KEY,
                mod_time REAL,
                DateTimeOriginal TEXT,
                Model TEXT,
                LensModel TEXT,
                ISO TEXT,
                ExposureTime TEXT,
                FNumber TEXT,
                FocalLength TEXT,
                Flash TEXT,
                WhiteBalance TEXT,
                ImageWidth TEXT,
                ImageHeight TEXT,
                FocalLengthIn35mmFormat TEXT
            )
        ''')
        conn.commit()  # Though context manager commits on exit, explicit commit also works

def get_db_connection(readonly: bool = False) -> sqlite3.Connection:
    """
    Get a database connection, optionally in read-only mode, with necessary PRAGMA settings.

    Args:
        readonly (bool): Open the database in read-only mode if True.

    Returns:
        sqlite3.Connection: The configured SQLite connection.
    """
    mode = "ro" if readonly else "rw"
    uri = f"file:{DB_FILE}?mode={mode}"
    try:
        conn = sqlite3.connect(uri, uri=True)
        if not readonly:
            # Apply improvements to write performance
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA cache_size = -10000")
        return conn
    except sqlite3.OperationalError as e:
        logging.error("Error connecting to database: %s", e)
        raise

def get_cached_metadata(conn: sqlite3.Connection, file_path: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached metadata for a specific file if it exists.

    Args:
        conn (sqlite3.Connection): The database connection to use.
        file_path (str): The full path of the file.

    Returns:
        Optional[Dict[str, Any]]: The metadata information or None if not present.
    """
    c = conn.cursor()
    c.execute('''
        SELECT mod_time, DateTimeOriginal, Model, LensModel, ISO, ExposureTime, FNumber,
               FocalLength, Flash, WhiteBalance, ImageWidth, ImageHeight, FocalLengthIn35mmFormat
        FROM metadata WHERE source_file=?
    ''', (file_path,))
    row = c.fetchone()
    if row:
        return {
            "mod_time": row[0],
            "DateTimeOriginal": row[1],
            "Model": row[2],
            "LensModel": row[3],
            "ISO": row[4],
            "ExposureTime": row[5],
            "FNumber": row[6],
            # Normalize focal length before returning
            "FocalLength": normalize_focal_length(row[7]),
            "Flash": row[8],
            "WhiteBalance": row[9],
            "ImageWidth": row[10],
            "ImageHeight": row[11],
            "FocalLengthIn35mmFormat": row[12]
        }
    return None

# ------------------------------------------------------------------------------
# Normalization Functions
# ------------------------------------------------------------------------------
def normalize_focal_length(focal: Optional[str]) -> str:
    """
    Normalize the focal length string for consistency.

    Args:
        focal (Optional[str]): The raw focal length string.

    Returns:
        str: Normalized focal length (e.g. "50 mm").
    """
    if not focal:
        return ""
    try:
        focal_clean = focal.lower().replace("mm", "").strip()
        value = float(focal_clean)
        # If value is an integer then display as int, else one decimal precision
        if value.is_integer():
            return f"{int(value)} mm"
        else:
            return f"{value:.1f} mm"
    except ValueError:
        return focal

def normalize_white_balance(wb: Optional[str]) -> str:
    """
    Normalize the white balance string.

    Args:
        wb (Optional[str]): The raw white balance string.

    Returns:
        str: A normalized white balance string or defaults to "manual" if invalid.
    """
    if not wb:
        return "manual"

    wb_norm = wb.strip().lower()
    if wb_norm.startswith("unknown") or wb_norm == "custom":
        return "manual"
    return WB_MAPPING.get(wb_norm, wb_norm)

# ------------------------------------------------------------------------------
# ExifTool Processing
# ------------------------------------------------------------------------------
def run_exiftool_batch(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Run ExifTool on a batch of files to extract metadata.

    Args:
        file_paths (List[str]): List of absolute file paths to process.

    Returns:
        List[Dict[str, Any]]: A list of metadata dictionaries.
    """
    try:
        result = subprocess.run(
            ["exiftool", "-json"] + file_paths,
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logging.error("ExifTool error on batch: %s", e)
        return []

# ------------------------------------------------------------------------------
# Multiprocessing Worker and Writer
# ------------------------------------------------------------------------------
def worker_process(files_chunk: List[str], writer_queue: multiprocessing.Queue, progress_queue: multiprocessing.Queue) -> None:
    """
    Process a chunk of files: decide which ones need metadata extraction
    and batch process them with ExifTool.

    Args:
        files_chunk (List[str]): A sublist of file paths.
        writer_queue (multiprocessing.Queue): Queue to send database insert items.
        progress_queue (multiprocessing.Queue): Queue to send progress updates.
    """
    try:
        # Open a read-only DB connection
        conn = get_db_connection(readonly=True)
        to_process: List[Tuple[str, float]] = []

        for file_path in files_chunk:
            abs_fp = os.path.abspath(file_path)
            # If file doesn't exist, inform monitor and skip it
            if not os.path.exists(abs_fp):
                progress_queue.put(('skip', abs_fp))
                continue

            mod_time = os.path.getmtime(abs_fp)
            cached = get_cached_metadata(conn, abs_fp)
            # If metadata is already cached and up to date, skip
            if cached and cached['mod_time'] == mod_time:
                progress_queue.put(('cached', abs_fp))
                continue

            # Mark for processing and notify progress
            to_process.append((abs_fp, mod_time))
            progress_queue.put(('process', abs_fp))

        conn.close()

        # Process files in batches
        for i in range(0, len(to_process), BATCH_SIZE):
            batch = to_process[i:i+BATCH_SIZE]
            batch_paths = [fp for fp, _ in batch]
            metadatas = run_exiftool_batch(batch_paths)

            # Pair each file with its metadata and push into the writer queue
            for (fp, mt), meta in zip(batch, metadatas):
                if 'Error' in meta:
                    logging.error("ExifTool error for %s: %s", fp, meta.get('Error'))
                    continue
                writer_queue.put((fp, mt, meta))
    except Exception as e:
        logging.error("Worker error: %s", e)

def writer_process(writer_queue: multiprocessing.Queue) -> None:
    """
    Retrieve metadata items from the writer queue and insert them into the database.

    Args:
        writer_queue (multiprocessing.Queue): Queue containing (file_path, mod_time, metadata) tuples.
    """
    try:
        conn = get_db_connection()
        count = 0
        while True:
            try:
                item = writer_queue.get(timeout=5)  # Wait for new item
                if item is None:
                    # None signals termination
                    break
                file_path, mod_time, metadata = item
                with conn:
                    # Insert or replace record in the database using the same schema
                    conn.execute('''
                        INSERT OR REPLACE INTO metadata VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    ''', (
                        file_path,
                        mod_time,
                        metadata.get("DateTimeOriginal"),
                        metadata.get("Model"),
                        metadata.get("LensModel"),
                        str(metadata.get("ISO")) if metadata.get("ISO") is not None else None,
                        str(metadata.get("ExposureTime")) if metadata.get("ExposureTime") is not None else None,
                        str(metadata.get("FNumber")) if metadata.get("FNumber") is not None else None,
                        normalize_focal_length(metadata.get("FocalLength")),
                        metadata.get("Flash"),
                        normalize_white_balance(metadata.get("WhiteBalance")),
                        metadata.get("ImageWidth"),
                        metadata.get("ImageHeight"),
                        metadata.get("FocalLengthIn35mmFormat")
                    ))
                count += 1
                # Commit in batches to reduce I/O overhead
                if count % DB_WRITE_BATCH_SIZE == 0:
                    conn.commit()
            except multiprocessing.queues.Empty:
                continue
            except Exception as e:
                logging.error("Writer error: %s", e)
        # Final commit and closing connection
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error("Writer setup error: %s", e)

def progress_monitor(total: int, progress_queue: multiprocessing.Queue) -> None:
    """
    Monitor the progress of file processing and display updates in the terminal.

    Args:
        total (int): Total number of files to process.
        progress_queue (multiprocessing.Queue): Queue containing progress events.
    """
    processed = 0
    start_time = time.time()

    while processed < total:
        try:
            status, file_path = progress_queue.get(timeout=1)
            processed += 1
            filename = os.path.basename(file_path)
            status_msg = "Cached" if status == 'cached' else "Processing"
            # Overwrite the current line in terminal with progress update
            sys.stdout.write(f"\r\033[K[{processed}/{total} ({processed/total:.1%})] {status_msg}: {filename}")
            sys.stdout.flush()
        except Exception:
            continue

    # Final print showing completion time
    sys.stdout.write(f"\r\033[K✅ Processing complete! {total} files processed in {format_time(time.time() - start_time)}\n")
    sys.stdout.flush()

# ------------------------------------------------------------------------------
# Directory and Statistics Functions
# ------------------------------------------------------------------------------
def process_directory(directory: str) -> Dict[Tuple[str, str], str]:
    """
    Recursively scan the provided directory, grouping files by their directory and base filename.
    RAW files take precedence over JPEG when duplicates exist.

    Args:
        directory (str): The directory path to scan.

    Returns:
        Dict[Tuple[str, str], str]: A dictionary where the key is a tuple (directory, basename)
                                    and the value is the file path.
    """
    directory = os.path.abspath(directory)
    grouped_files: Dict[Tuple[str, str], str] = {}

    for root, _, files in os.walk(directory):
        local_group: Dict[Tuple[str, str], str] = {}
        for f in files:
            ext = f.split('.')[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue

            full_path = os.path.join(root, f)
            base_name = os.path.splitext(f)[0]
            key = (root, base_name)

            # Update local grouping: RAWs have priority over JPEGs
            if key in local_group:
                existing_ext = os.path.splitext(local_group[key])[1][1:].lower()
                if existing_ext in JPEG_EXTENSIONS and ext in RAW_EXTENSIONS:
                    local_group[key] = full_path
            else:
                local_group[key] = full_path

        # Merge local group into final grouping dictionary
        for key, path in local_group.items():
            if key in grouped_files:
                existing_ext = os.path.splitext(grouped_files[key])[1][1:].lower()
                new_ext = os.path.splitext(path)[1][1:].lower()
                if existing_ext in JPEG_EXTENSIONS and new_ext in RAW_EXTENSIONS:
                    grouped_files[key] = path
            else:
                grouped_files[key] = path

    return grouped_files

def generate_statistics(grouped_files: Dict[Tuple[str, str], str]) -> Dict[str, Counter]:
    """
    Generate various counters based on metadata from the database for groups of files.

    Args:
        grouped_files (Dict[Tuple[str, str], str]): Dictionary of grouped file paths.

    Returns:
        Dict[str, Counter]: Dictionary of counters for each category (e.g., year, camera, etc.).
    """
    conn = get_db_connection(readonly=True)
    counters: Dict[str, Counter] = {
        'year': Counter(),
        'month': Counter(),
        'camera': Counter(),
        'lens': Counter(),
        'iso': Counter(),
        'shutter': Counter(),
        'aperture': Counter(),
        'focal': Counter(),
        'flash': Counter(),
        'white_balance': Counter(),
        'resolution': Counter(),
        'focal35': Counter()
    }

    for file_path in grouped_files.values():
        abs_fp = os.path.abspath(file_path)
        meta = get_cached_metadata(conn, abs_fp)
        if not meta:
            continue

        date_str = meta.get("DateTimeOriginal")
        if date_str:
            try:
                # Convert string to datetime; expected format: "YYYY:MM:DD HH:MM:SS"
                dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                counters['year'][dt.year] += 1
                counters['month'][dt.month] += 1
            except ValueError:
                pass

        if model := meta.get("Model"):
            counters['camera'][model.strip()] += 1
        if lens := meta.get("LensModel"):
            counters['lens'][lens.strip()] += 1
        if iso := meta.get("ISO"):
            counters['iso'][str(iso)] += 1
        if shutter := meta.get("ExposureTime"):
            counters['shutter'][str(shutter)] += 1
        if aperture := meta.get("FNumber"):
            counters['aperture'][str(aperture)] += 1
        if focal := meta.get("FocalLength"):
            counters['focal'][normalize_focal_length(focal)] += 1
        if flash := meta.get("Flash"):
            counters['flash'][str(flash)] += 1
        if wb := meta.get("WhiteBalance"):
            counters['white_balance'][normalize_white_balance(wb)] += 1
        if width := meta.get("ImageWidth"):
            if height := meta.get("ImageHeight"):
                counters['resolution'][f"{width}x{height}"] += 1
        if focal35 := meta.get("FocalLengthIn35mmFormat"):
            counters['focal35'][str(focal35)] += 1

    conn.close()
    return counters

def print_counter(title: str, counter: Counter, formatter=lambda x: x, threshold: int = 3) -> None:
    """
    Print the counter statistics in a formatted manner, grouping low-frequency items into "Other".

    Args:
        title (str): Title for the counter.
        counter (Counter): Counter object.
        formatter (callable): Function to convert counter key to a formatted string.
        threshold (int): Minimum frequency to show individually.
    """
    main_items = {k: v for k, v in counter.items() if v >= threshold}
    other_total = sum(v for k, v in counter.items() if v < threshold)
    print(f"=== {title} ===")
    for item, count in sorted(main_items.items(), key=lambda x: (-x[1], x[0])):
        print(f"{formatter(item)}: {count}")
    if other_total > 0:
        print(f"Other (<{threshold}): {other_total}")
    print()

# ------------------------------------------------------------------------------
# Main Processing Function
# ------------------------------------------------------------------------------
def main() -> None:
    """
    Main function that executes the flow of the script:
     - Print credits and select directory
     - Ensure database table exists
     - Recursively scan directory, and prepare file list
     - Start multiprocessing for reading metadata and database writing
     - Generate and display statistics
    """
    print(CREDITS)
    # Use command-line arg or default to current working directory
    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    print(f"📂 Processing directory: {directory}")

    try:
        start_time = time.time()
        # Ensure database exists with correct table schema
        create_tables_if_needed()

        print("🔍 Scanning directory structure.")
        grouped_files = process_directory(directory)
        file_list = list(grouped_files.values())
        total_photos = len(file_list)
        print(f"📷 Found {total_photos} photos to process")

        if not file_list:
            print("🚫 No photos found")
            return

        # Set up multiprocessing resources
        manager = Manager()
        writer_queue = manager.Queue()
        progress_queue = manager.Queue()
        num_workers = os.cpu_count() or 4

        print("🚀 Starting metadata processing.")
        # Launch writer and progress monitor processes
        writer = Process(target=writer_process, args=(writer_queue,))
        monitor = Process(target=progress_monitor, args=(total_photos, progress_queue))
        writer.start()
        monitor.start()

        # Determine chunk size dynamically; tweak as needed for optimal performance
        chunk_size = max(1, len(file_list) // (num_workers * 2))
        chunks = [file_list[i:i+chunk_size] for i in range(0, len(file_list), chunk_size)]

        # Use multiprocessing pool to process file chunks
        with Pool(num_workers) as pool:
            pool.starmap(worker_process, [(chunk, writer_queue, progress_queue) for chunk in chunks])

        # Signal the writer process to exit by sending None
        writer_queue.put(None)
        writer.join()
        monitor.join()

        print("\n📈 Generating statistics.")
        counters = generate_statistics(grouped_files)

        # Display various summary statistics
        print_counter("Year Statistics", counters['year'], lambda y: f"Year {y}")
        print_counter("Month Statistics", counters['month'], lambda m: f"Month {m}")
        print_counter("Camera Models", counters['camera'])
        print_counter("Lens Models", counters['lens'])
        print_counter("ISO Statistics", counters['iso'], lambda iso: f"ISO {iso}")
        print_counter("Shutter Speed", counters['shutter'], lambda s: f"{s}s")
        print_counter("Aperture", counters['aperture'], lambda a: f"f/{a}")
        print_counter("Focal Length", counters['focal'])
        print_counter("Flash Usage", counters['flash'])
        print_counter("White Balance", counters['white_balance'])
        print_counter("Resolution", counters['resolution'])
        print_counter("35mm Focal Length", counters['focal35'])

        total_time = time.time() - start_time
        print(f"\n⌛ Total processing time: {format_time(total_time)}")
        print(CREDITS)

    except KeyboardInterrupt:
        print("\n🛑 Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.exception("Fatal error")
        print(f"\n❌ Error occurred: {e}\n🔍 See {ERROR_LOG} for details")
        sys.exit(1)

# ------------------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    main()
