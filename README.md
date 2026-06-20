# HN1 Load Retimer

Removes loading screens and retimes a run for HN1.

## Download

Grab the latest release from the [Releases](../../releases) page it includes
the compiled exe and all dependencies. No Python installation needed.

## Running from source

1. Install [Python 3.9+](https://www.python.org/downloads/)
2. Download this repo
3. Install dependencies:
   ```
   pip install -U numpy pillow scipy customtkinter
   ```
4. Run:
   ```
   python retime.py
   ```

## Building from source

1. Install [Python 3.9+](https://www.python.org/downloads/)
2. Install dependencies:
   ```
   pip install -U numpy pillow scipy customtkinter pyinstaller
   ```
3. Place all dependency files into `dependencies/`:
   - `1.png`, `2.png`, `3.png`, `icon.ico`
   - `ffmpeg.exe`, `ffprobe.exe`, `yt-dlp.exe`, `deno.exe`
4. Build:
   ```
   pyinstaller --noconsole --onefile --icon=dependencies/icon.ico retime.py
   ```
5. The compiled exe will be at `dist/retime.exe`. Place it next to the `dependencies/` folder to use.

## Binaries (.exe dependencies)

This project requires external binaries for video downloading, processing, and frame detection. While pre-compiled versions are included in the repository, you can download the official, up-to-date versions directly from the sources below:

| Dependency | Purpose | Download Link |
| :--- | :--- | :--- |
| **`ffmpeg.exe`** | Video processing & frame extraction | [FFmpeg Official Download](https://ffmpeg.org/download.html) (Windows Build) |
| **`ffprobe.exe`** | Video metadata analysis | *Included with the FFmpeg download* |
| **`yt-dlp.exe`** | Video downloading | [yt-dlp Releases](https://github.com/yt-dlp/yt-dlp/releases) |
| **`deno.exe`** | JavaScript/TypeScript runtime | [Deno Official Site](https://deno.land/) |

## Usage

- Paste a YouTube link or browse for a local video file
- Choose **Fullscreen run** or **Windowed run**
  - Windowed: click *Select game area* and drag a box around the game window
- Set start and end times (or paste from clipboard)
- Click **Retime**

Downloaded videos are saved to a `Videos/` folder next to the exe.

## License

MIT
