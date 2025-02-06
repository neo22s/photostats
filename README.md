# Photo Stats

**Photo Stats** is a Python script designed for photographers to analyze their photo metadata (EXIF) and generate detailed statistics about their work. The script scans all files (RAW and JPEG) in a given directory and its subdirectories, groups them by folder and base filename (preferring RAW files when available), and extracts key metadata fields. It then stores these values in a SQLite database to cache results for faster subsequent processing.

## Features

- **Recursive File Scanning:**
  The script scans the specified folder and all its subfolders for images with allowed extensions (RAW and JPEG).

- **Grouping & Deduplication:**
  Files are grouped by their folder and base filename. If both a RAW and a JPEG version exist for the same image, the RAW file is preferred.

- **Metadata Extraction:**
  Extracts key EXIF metadata such as:
  - DateTimeOriginal
  - Camera Model
  - Lens Model
  - ISO
  - Shutter Speed (ExposureTime)
  - Aperture (FNumber)
  - Focal Length
  - Flash status
  - White Balance
  - Image Resolution (ImageWidth x ImageHeight)
  - Focal Length (35mm Format)

- **SQLite Caching:**
  To speed up subsequent runs, the script stores the extracted metadata in a SQLite database (`photo_stats_cache.db`). New files (i.e., files with new absolute paths) are added automatically.

- **Statistics Generation:**
  It aggregates the metadata into statistics (e.g., the number of photos taken each year, by camera, by lens, etc.) and displays them in a clear format.

## Requirements

- **Python 3.x**
  Ensure Python 3 is installed on your system.

- **ExifTool**
  The script relies on [ExifTool](https://exiftool.org/) to extract metadata. Make sure it is installed and accessible via the command line.

- **SQLite3**
  Python's built-in `sqlite3` module is used to manage the cache database.

## Installation

### macOS / Linux

1. **Install Python 3 (if not already installed):**
   You can install Python 3 from your package manager or download it from [python.org](https://www.python.org/downloads/).

2. **Install ExifTool:**
   - On macOS using Homebrew:
     ```bash
     brew install exiftool
     ```
   - On Linux, you can install ExifTool via your package manager (e.g., `sudo apt install libimage-exiftool-perl` on Debian/Ubuntu).

3. **Clone this Repository:**
   ```bash
   git clone https://github.com/neo22s/photostats.git
   cd photostats
   ```

4. **Run the Script:**
   ```bash
   python3 photo_stats.py /path/to/your/photos
   ```
   If you omit the path, the script uses the current directory.

### Windows

1. **Install Python 3:**
   Download and install Python 3 from [python.org](https://www.python.org/downloads/windows/). During installation, check the box to add Python to your PATH.

2. **Install ExifTool:**
   - Download `exiftool.exe` from [ExifTool's website](https://exiftool.org/).
   - Optionally, rename it to `exiftool.exe` and place it in a folder that's in your system PATH (e.g., `C:\Windows\`).

3. **Clone this Repository:**
   ```batch
   git clone https://github.com/neo22s/photostats.git
   cd photostats
   ```

4. **Run the Script:**
   ```batch
   python photo_stats.py "C:\path\to\your\photos"
   ```
   If you omit the path, the script uses the current directory.

## How It Works

1. **Scanning and Grouping:**
   The script recursively scans the given directory and groups files by their folder and base filename. This ensures that if both a RAW and a JPEG file exist for the same photo, only the RAW file is processed.

2. **Metadata Extraction:**
   Using ExifTool, the script extracts metadata from each image. The data is normalized (e.g., focal length values are formatted uniformly, white balance values are grouped).

3. **Caching:**
   All metadata is stored in a SQLite database. The script checks for existing entries by file absolute path. If an entry exists, it is used; otherwise, the file is processed and added to the DB.

4. **Statistics Generation:**
   Finally, the script aggregates statistics from the metadata (e.g., photos per year, by camera, by lens, etc.) and prints the results in the terminal.

## Credits

Developed by **@chema_photo** – Follow me on [Instagram](https://www.instagram.com/chema_photo/) and [YouTube](https://www.youtube.com/@ChemaPhoto)!
More info about the script can be found at [chemaphoto.com](https://chemaphoto.com).


Feel free to open issues or contribute to the project. Happy coding and happy shooting!
