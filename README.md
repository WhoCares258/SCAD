# SCAD: Surveillance Curated Accident Dataset

**A cleaner benchmark for accident recognition in surveillance videos.**

Accidents are rare, brief, and often difficult to distinguish from ordinary traffic activity. **SCAD** brings together carefully selected surveillance footage from four established datasets to support reproducible research on **normal vs. accident/anomalous video classification**. By curating duplicate footage and keeping the original evaluation splits fixed, SCAD provides a practical benchmark for comparing video recognition models under consistent conditions.

SCAD contains **1,122 four-second clips**, balanced across **561 anomalous** and **561 normal** samples. The fixed split is **778 training**, **112 validation**, and **232 test** clips. Every clip is encoded at **30 FPS (120 frames)**. The source datasets are **CADP, SO-TAD, TU-DAT, and UCF-Crime**.

## Research papers

**Published foundation paper (ETASR, June 2026)**  
[From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNNs](https://doi.org/10.48084/etasr.17024)

This paper introduces the SCAD curation and benchmarking approach and establishes its initial accident-recognition baselines.

**Follow-up study (in progress, target release: early 2027)**  
*Benchmarking Video Encoders for Surveillance Accident Recognition*  
**Paper link:** Coming with the release. This study extends the benchmark to a broader comparison of modern video encoders.

## Reconstruct SCAD

The repository contains the reconstruction script and the manifest specifying the exact clip names, labels, and train/validation/test assignments. **The original source videos must be downloaded separately** from their respective providers.

### 1. Clone this repository and check the requirements

Install [Git](https://git-scm.com/), [Python 3.9+](https://www.python.org/downloads/), [FFmpeg](https://ffmpeg.org/download.html) (including `ffprobe`), and [7-Zip](https://www.7-zip.org/). Make sure `python`, `ffmpeg`, and `ffprobe` can be called from a terminal.

Open **PowerShell** in the folder where you want to store the datasets:

```powershell
git clone https://github.com/WhoCares258/SCAD.git
cd SCAD
```

**Important:** All commands below are run **inside the cloned `SCAD` folder**. The four source-data folders should be placed **one level above it**, as shown in the final layout.

### 2. TU-DAT (easiest)

1. Open the [TU-DAT original manuscript](https://www.mdpi.com/1424-8220/25/11/3259). Read the dataset-access instructions in the section below the abstract and follow its dataset link. You can also open the [source Google Drive folder](https://drive.google.com/drive/folders/1qsORgV4p4yn-hxsVFYKdRZePYL2m_ncj) directly.
2. Find **`Positive_Vidoes`**. Click the **three-dot (⋮) menu** next to the folder and select **Download**.
3. Google Drive may spend several minutes preparing a ZIP before the download begins. Download and extract it.
4. Move the extracted **`Positive_Vidoes`** folder beside `SCAD`, not inside it. Keep the original spelling, including `Vidoes`.

You should have files such as `../Positive_Vidoes/v14.mov`.

### 3. UCF-Crime (download one archive only)

1. Open the [UCF-Crime official archive on Dropbox](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa).
2. Download **`Anomaly-Videos-Part-3.zip` only**. You do not need the other UCF-Crime archives for SCAD.
3. Extract the ZIP, locate the **`RoadAccidents`** folder, and move it beside `SCAD`.

You should have files such as `../RoadAccidents/RoadAccidents013_x264.mp4`.

### 4. SO-TAD (Hugging Face download)

Original project: [SO-TAD GitHub](https://github.com/cccxy-299/so-tad)  
Dataset files: [SO-TAD on Hugging Face](https://huggingface.co/datasets/cccccxy/so-tad)

1. Install the Hugging Face command-line tool:

   ```powershell
   python -m pip install -U huggingface_hub
   ```

2. From the cloned `SCAD` directory, download the dataset directly into a sibling folder:

   ```powershell
   hf download cccccxy/so-tad --repo-type dataset --local-dir ..\SO-TAD
   ```

3. After the download completes, locate the multipart archive (`so_tad.zip` together with `so_tad.z01`, `so_tad.z02`, and so on). **Keep all parts in one folder.** Using 7-Zip, extract from **`so_tad.zip`**; do not extract each `.z01` file individually.
4. Arrange the extracted videos so that the paths are `../SO-TAD/so_tad/train/*.mp4` and `../SO-TAD/so_tad/test/*.mp4`. If extraction adds an extra outer directory, move the inner `so_tad` folder into place.

**Duplicate note:** The available SO-TAD release removed source `106` as a duplicate of source `47`. In SCAD, `47_1.mp4` is retained from segment 1, while the retained `106_2.mp4` entry is rebuilt using segment `47_2`. The original filename, class, and split are preserved.

### 5. CADP (download only the required frame folders)

Project and dataset instructions: [CADP](https://ankitshah009.github.io/accident_forecasting_traffic_camera)  
[Official extracted JPEG frames on Google Drive](https://drive.google.com/drive/folders/1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v)

CADP supplies numbered JPEG frames in source-specific folders. You only need the folders referenced by SCAD, not every CADP sequence.

1. Read the CADP dataset instructions and install [rclone](https://rclone.org/downloads/). We use **rclone** for selective Google Drive downloading.
2. For better Google Drive quota handling, optionally [create your own Google API client ID and secret](https://rclone.org/drive/#making-your-own-client-id). Keep these credentials private. The default rclone Google Drive client is also an option.
3. In PowerShell, configure a Google Drive remote:

   ```powershell
   rclone config
   ```

   Create a new remote named **`cadp`**, choose **Google Drive** as its type, enter your client ID/secret if you created them, and complete browser authorization. For the **root folder ID**, use:

   ```text
   1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v
   ```

   This points the remote at the CADP frame collection. See the [rclone Google Drive setup guide](https://rclone.org/drive/) if the available options differ.

4. Generate a download filter directly from SCAD's manifest. This selects only the required CADP source folders:

   ```powershell
   Import-Csv .\manifests\scad_manifest.csv |
       Where-Object { $_.dataset -eq 'CADP' } |
       Select-Object -ExpandProperty source_id -Unique |
       ForEach-Object { "$_/**" } |
       Set-Content .\manifests\cadp_include.txt
   ```

5. Download the selected JPEG folders **from inside the cloned SCAD directory**:

   ```powershell
   rclone copy cadp: ..\CADP --include-from .\manifests\cadp_include.txt -P
   ```

   If the Google Drive owner applies access or quota restrictions, follow their published access requirements; your own API client may help with shared-client rate limits but does not bypass owner restrictions.

You should have folders such as `../CADP/000002/0.jpg`, `../CADP/000002/1.jpg`, and so on.

## Final folder layout

Before reconstruction, check that your files are arranged as follows:

```text
repository/
├── SCAD/                            # Cloned GitHub repository
│   ├── README.md
│   ├── scripts/
│   │   └── reconstruct_scad.py
│   └── manifests/
│       └── scad_manifest.csv
├── Positive_Vidoes/                 # TU-DAT
│   ├── v14.mov
│   └── ...
├── RoadAccidents/                   # UCF-Crime
│   ├── RoadAccidents013_x264.mp4
│   └── ...
├── SO-TAD/                          # SO-TAD
│   └── so_tad/
│       ├── train/
│       └── test/
└── CADP/                            # CADP
    ├── 000002/
    │   ├── 0.jpg
    │   └── 1.jpg
    └── ...
```

## Generate the dataset

From **inside `SCAD`**, run:

```powershell
python .\scripts\reconstruct_scad.py
```

To replace previously generated clips, use:

```powershell
python .\scripts\reconstruct_scad.py --overwrite
```

The output is created at `SCAD/SCAD_output/` with the original class and split folders:

```text
SCAD_output/
├── train/
│   ├── anomalous/
│   └── normal/
├── val/
│   ├── anomalous/
│   └── normal/
└── test/
    ├── anomalous/
    └── normal/
```

The reconstruction uses the **exact 1,122 entries in `manifests/scad_manifest.csv`**. The clips are four seconds long, sampled with a three-second stride, encoded at 30 FPS, and padded with the final frame where required. The script uses the appropriate source processing for each dataset: JPEG sequence reconstruction for CADP; frame-preserving 30 FPS playback for SO-TAD and TU-DAT; and direct MP4 segmentation for UCF-Crime.

**Verification:** All 1,122 SCAD clips have been verified using SSIM comparison or manual visual inspection. The provided reconstruction script produces the dataset without requiring the original SCAD reference files or running SSIM checks.

## Use and citation

Please cite the [SCAD foundation paper](https://doi.org/10.48084/etasr.17024) and the relevant original source datasets when using this benchmark. Source material remains subject to each provider's own access and license terms. This repository distributes reconstruction code and the clip manifest, **not the original video footage**.
