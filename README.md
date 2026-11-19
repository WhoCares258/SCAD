# SCAD: Surveillance Curated Accident Dataset

**A curated, reproducible benchmark for accident recognition in surveillance videos.**

SCAD brings together footage from CADP, SO-TAD, TU-DAT, and UCF-Crime for **normal versus accident/anomalous video classification**. Duplicate removal and fixed evaluation splits support consistent model comparisons. It contains **1,122 four-second clips** (561 anomalous and 561 normal), split into **778 training, 112 validation, and 232 test** clips, each at **30 FPS (120 frames)**.

## Research papers

**Dataset curation and initial benchmarking (ETASR, June 2026)**  
[From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNNs](https://doi.org/10.48084/etasr.17024)

**Follow-up video encoder benchmarking study (in review; target release: early 2027)**  
*Benchmarking Video Encoders for Surveillance Accident Recognition*  
Paper link: **Coming soon**.

## Reconstruct SCAD

The repository provides the reconstruction script and the manifest defining the exact clips, class labels, and splits. **Download the original source footage separately** from the respective dataset providers.

**Requirements:** [Python 3.9+](https://www.python.org/downloads/), [FFmpeg](https://ffmpeg.org/download.html) (including `ffprobe`), [Git](https://git-scm.com/), and [7-Zip](https://www.7-zip.org/). Ensure `python`, `ffmpeg`, and `ffprobe` are available in your terminal.

### Clone the repository

```powershell
git clone https://github.com/WhoCares258/SCAD.git
cd SCAD
```

Run the following commands **inside the cloned `SCAD` folder**. Place the four source-data folders **beside `SCAD`**, one directory above it.

### 1. TU-DAT

1. Read the **dataset-access instructions below the abstract** in the [original TU-DAT paper](https://www.mdpi.com/1424-8220/25/11/3259), then open the [dataset Google Drive folder](https://drive.google.com/drive/folders/1qsORgV4p4yn-hxsVFYKdRZePYL2m_ncj).
2. Click the **three-dot menu** beside `Positive_Vidoes` and select **Download**. Google Drive may take a few minutes to prepare the ZIP before downloading.
3. Extract the archive and move `Positive_Vidoes` **beside** the cloned `SCAD` folder, retaining its original spelling. The videos should be at `../Positive_Vidoes/*.mov`.

### 2. UCF-Crime

1. Open the [official UCF-Crime Dropbox archive](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa).
2. Download **`Anomaly-Videos-Part-3.zip` only**; the other parts are not needed for SCAD.
3. Extract the ZIP and move its `RoadAccidents` folder beside `SCAD`, giving `../RoadAccidents/*.mp4`.

### 3. SO-TAD

Original project: [GitHub](https://github.com/cccxy-299/so-tad) | Dataset: [Hugging Face](https://huggingface.co/datasets/cccccxy/so-tad)

1. Install the Hugging Face command-line tool in a terminal opened inside `SCAD`:

   ```powershell
   python -m pip install -U huggingface_hub
   ```

2. Download the dataset directly into the correct sibling directory:

   ```powershell
   hf download cccccxy/so-tad --repo-type dataset --local-dir ..\SO-TAD
   ```

3. In `../SO-TAD`, keep all archive parts (`so_tad.zip` and `.z01`, `.z02`, etc.) together. Right-click **`so_tad.zip`**, select **7-Zip → Extract**, and extract the multipart archive once. Do not extract the `.z01` parts individually.
4. Ensure the extracted video folders are `../SO-TAD/so_tad/train/` and `../SO-TAD/so_tad/test/`. Remove any extra enclosing folder created during extraction.

**Duplicate note:** The available SO-TAD release removed source `106` as a duplicate of `47`. SCAD retains `47_1.mp4`, while its retained `106_2.mp4` entry is reconstructed from source 47, segment 2 (`47_2`), preserving the SCAD filename, label, and split.

### 4. CADP

Original project and access instructions: [CADP](https://ankitshah009.github.io/accident_forecasting_traffic_camera) | [JPEG frame collection](https://drive.google.com/drive/folders/1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v)

CADP contains numbered JPEG frames. The provided **`manifests/cadp_include.txt`** selects only the source folders required by SCAD.

1. Read the CADP project's dataset instructions and install [rclone](https://rclone.org/downloads/).
2. For faster and more reliable Google Drive access, consider creating a [personal Google API client ID and secret](https://rclone.org/drive/#making-your-own-client-id). This is optional; keep credentials private. The shared default client may be slower or subject to rate limits.
3. Configure rclone in PowerShell:

   ```powershell
   rclone config
   ```

   Create a new remote named **`cadp`**, choose **Google Drive**, enter your own API credentials if available, and complete browser authorization. In advanced settings, set **`root_folder_id`** to:

   ```text
   1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v
   ```

   Refer to the [rclone Google Drive guide](https://rclone.org/drive/) if your configuration screens differ.

4. From inside `SCAD`, check that the supplied include file exists, then download only the selected CADP folders:

   ```powershell
   Test-Path .\manifests\cadp_include.txt
   rclone copy cadp: ..\CADP --include-from .\manifests\cadp_include.txt -P
   ```

   `Test-Path` should return `True`. The JPEGs will be downloaded to folders such as `../CADP/000002/0.jpg`. Your API client does not bypass the source owner's permissions or Google Drive quotas.

## Final folder layout

Before reconstruction, check the following structure:

```text
repository/
├── SCAD/                             # Cloned GitHub repository
│   ├── README.md
│   ├── scripts/
│   │   └── reconstruct_scad.py
│   └── manifests/
│       ├── scad_manifest.csv
│       └── cadp_include.txt
├── Positive_Vidoes/                  # TU-DAT
│   ├── v14.mov
│   └── ...
├── RoadAccidents/                    # UCF-Crime
│   ├── RoadAccidents013_x264.mp4
│   └── ...
├── SO-TAD/
│   └── so_tad/
│       ├── train/
│       └── test/
└── CADP/
    ├── 000002/
    │   ├── 0.jpg
    │   └── 1.jpg
    └── ...
```

## Generate the dataset

From inside the cloned `SCAD` directory, run:

```powershell
python .\scripts\reconstruct_scad.py
```

To regenerate existing outputs:

```powershell
python .\scripts\reconstruct_scad.py --overwrite
```

The script builds all **1,122 manifest-defined clips** in `SCAD_output/train/`, `SCAD_output/val/`, and `SCAD_output/test/`, each with `anomalous/` and `normal/` folders. Clips are four seconds at 30 FPS, extracted every three seconds and end-padded when needed. Dataset-specific frame processing is handled automatically.

**Verification:** All 1,122 clips have been verified through SSIM comparison or manual visual inspection. The public script does not require reference SCAD videos or perform SSIM checks.

## Use and citation

**Research and non-commercial use only. Do not redistribute the downloaded source footage or reconstructed SCAD video clips.** Download the originals from their respective providers and follow each provider's own access and usage terms. This repository distributes reconstruction instructions, code, and manifests only; it does not grant additional rights to third-party videos.

Please cite the SCAD dataset paper **and** the four contributing datasets:

1. **SCAD:** M. U. Danka, R. S. S. Singh, and M. A. Danka, “From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNNs,” *Engineering, Technology & Applied Science Research*, vol. 16, no. 3, pp. 36051–36057, 2026. https://doi.org/10.48084/etasr.17024
2. **TU-DAT:** P. P. Kumar and K. Kant, “TU-DAT: A Computer Vision Dataset on Road Traffic Anomalies,” *Sensors*, vol. 25, no. 11, art. 3259, 2025. https://doi.org/10.3390/s25113259
3. **UCF-Crime:** W. Sultani, C. Chen, and M. Shah, “Real-World Anomaly Detection in Surveillance Videos,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 6479–6488, 2018. https://doi.org/10.1109/CVPR.2018.00678
4. **SO-TAD:** X. Chen, H. Xu, M. Ruan, M. Bian, Q. Chen, and Y. Huang, “SO-TAD: A Surveillance-Oriented Benchmark for Traffic Accident Detection,” *Neurocomputing*, vol. 618, art. 129061, 2025. https://doi.org/10.1016/j.neucom.2024.129061
5. **CADP:** A. P. Shah, J.-B. Lamare, T. Nguyen-Anh, and A. Hauptmann, “CADP: A Novel Dataset for CCTV Traffic Camera Based Accident Analysis,” in *Proceedings of the 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)*, pp. 1–9, 2018. https://doi.org/10.1109/AVSS.2018.8639160
