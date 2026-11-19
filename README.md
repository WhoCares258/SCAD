# SCAD: Surveillance Curated Accident Dataset

**A curated benchmark for surveillance-based accident recognition.**

SCAD combines footage from **CADP, SO-TAD, TU-DAT, and UCF-Crime** for binary **normal vs. accident/anomalous** video classification. It contains **1,122 balanced four-second clips** (561 per class), with fixed **train/validation/test splits of 778/112/232**. All clips are **30 FPS (120 frames)**, supporting reproducible comparisons across video models.

## Research papers

**Dataset curation and initial benchmarking** (ETASR, June 2026)  
[From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNNs](https://doi.org/10.48084/etasr.17024)

**Follow-up video encoder benchmarking study** (in review; target release: early 2027)  
*Benchmarking Video Encoders for Surveillance Accident Recognition*  
Paper link: **Coming soon**.

## Reconstruct SCAD

This repository provides the reconstruction script and manifest defining the exact clips, labels, and splits. **Download the original footage separately** from the source providers.

**Requirements:** [Python 3.9+](https://www.python.org/downloads/), [FFmpeg and ffprobe](https://ffmpeg.org/download.html), [Git](https://git-scm.com/), and [7-Zip](https://www.7-zip.org/). Ensure `python`, `ffmpeg`, and `ffprobe` are available from your terminal.

### Clone the repository

```powershell
git clone https://github.com/WhoCares258/SCAD.git
cd SCAD
```

**All four source folders go inside the cloned `SCAD` folder.** Run the commands below from that folder.

### 1. TU-DAT

1. Read the **dataset-access instructions below the abstract** in the [TU-DAT paper](https://www.mdpi.com/1424-8220/25/11/3259), then open the [Google Drive dataset folder](https://drive.google.com/drive/folders/1qsORgV4p4yn-hxsVFYKdRZePYL2m_ncj).
2. Click the **three-dot menu** beside `Positive_Vidoes` and choose **Download**. Google Drive may take several minutes to prepare the ZIP.
3. Extract the archive and place the `Positive_Vidoes` folder in `SCAD/`, keeping its original spelling. Videos should appear as `Positive_Vidoes/*.mov`.

### 2. UCF-Crime

1. Open the [official UCF-Crime Dropbox archive](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa).
2. Download **`Anomaly-Videos-Part-3.zip` only**. The other archives are unnecessary for SCAD.
3. Extract the archive and move its `RoadAccidents` folder into `SCAD/`, so videos appear as `RoadAccidents/*.mp4`.

### 3. SO-TAD

Original project: [GitHub](https://github.com/cccxy-299/so-tad) | Dataset: [Hugging Face](https://huggingface.co/datasets/cccccxy/so-tad)

1. Install the Hugging Face command-line tool:

   ```powershell
   python -m pip install -U huggingface_hub
   ```

2. Download directly into the cloned repository:

   ```powershell
   hf download cccccxy/so-tad --repo-type dataset --local-dir .\SO-TAD
   ```

3. Inside `SO-TAD`, keep `so_tad.zip` and all `.z01`, `.z02`, etc. parts together. Right-click **`so_tad.zip`**, choose **7-Zip → Extract**, and extract the multipart archive once. Do not extract the numbered parts separately.
4. Confirm the video paths are `SO-TAD/so_tad/train/*.mp4` and `SO-TAD/so_tad/test/*.mp4`.

**Duplicate note:** During the initial SCAD curation, `47_2.mp4` and `106_2.mp4` were identified as duplicates, and `47_2.mp4` was removed while `106_2.mp4` was retained. However, the available SO-TAD release subsequently removed source `106` as a duplicate of source `47`. Therefore, reconstruction uses source `47`, segment 2 (`47_2`), to reproduce the retained `106_2.mp4` entry, preserving its original filename, label, and split. `47_1.mp4` remains unchanged.

### 4. CADP

Project and access instructions: [CADP](https://ankitshah009.github.io/accident_forecasting_traffic_camera) | [Original JPEG frame collection](https://drive.google.com/drive/folders/1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v)

CADP provides numbered JPEG image sequences. The included **`manifests/cadp_include.txt`** selects only the **168 source folders** needed by SCAD.

1. Read the CADP project's access instructions and install [rclone](https://rclone.org/downloads/).
2. To speed up downloads and reduce shared-client rate limits, [create your own Google API client ID and secret](https://rclone.org/drive/#making-your-own-client-id) and use them when configuring rclone. Keep these credentials private.
3. Run `rclone config`. Create a Google Drive remote named **`cadp`**, add your personal credentials if available, and complete browser authorization. Under advanced settings set **`root_folder_id`** to `1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v`. See the [rclone Google Drive guide](https://rclone.org/drive/) if prompted differently.
4. From the cloned `SCAD` folder, verify the include file and download the selected folders:

   ```powershell
   Test-Path .\manifests\cadp_include.txt
   rclone copy cadp: .\CADP --include-from .\manifests\cadp_include.txt -P
   ```

   `Test-Path` should show `True`. Frames will appear as `CADP/000002/0.jpg`, etc. A personal API client does not bypass Google Drive access restrictions or the source owner's permissions.

## Final folder layout

Once downloaded, your cloned repository should look like:

```text
SCAD/
├── README.md
├── scripts/
│   └── reconstruct_scad.py
├── manifests/
│   ├── scad_manifest.csv
│   └── cadp_include.txt
├── Positive_Vidoes/           # TU-DAT
│   ├── v14.mov
│   └── ...
├── RoadAccidents/             # UCF-Crime
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

From inside `SCAD`, run:

```powershell
python .\scripts\reconstruct_scad.py
```

To regenerate existing clips:

```powershell
python .\scripts\reconstruct_scad.py --overwrite
```

The script reconstructs all **1,122 SCAD clips**, preserving their original labels and dataset splits.

All clips have been successfully reconstructed and verified through SSIM comparison or manual visual inspection. **SCAD is now ready for reproducible research and benchmarking.**

## Use and citation

**Academic and non-commercial research use only. Do not redistribute the original source footage or reconstructed SCAD video clips.** Obtain footage from its respective providers and comply with each provider's access and usage terms. This repository publishes reconstruction code, instructions, and manifests, not third-party videos. It does not grant rights beyond those provided by the original dataset owners.

Please cite **SCAD and all four contributing datasets**:

### SCAD

```bibtex
@article{danka2026scad,
  author  = {Danka, Muhammad Umer and Singh, Ranjit Singh Sarban and Danka, Muhammad Ayoub},
  title   = {From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNNs},
  journal = {Engineering, Technology \& Applied Science Research},
  volume  = {16},
  number  = {3},
  pages   = {36051--36057},
  year    = {2026},
  doi     = {10.48084/etasr.17024}
}
```

### TU-DAT

```bibtex
@article{pradeepkumar2025tudat,
  author  = {Pradeep Kumar, Pavana and Kant, Krishna},
  title   = {{TU-DAT}: A Computer Vision Dataset on Road Traffic Anomalies},
  journal = {Sensors},
  volume  = {25},
  number  = {11},
  pages   = {3259},
  year    = {2025},
  doi     = {10.3390/s25113259}
}
```

### UCF-Crime

```bibtex
@inproceedings{sultani2018ucfcrime,
  author    = {Sultani, Waqas and Chen, Chen and Shah, Mubarak},
  title     = {Real-World Anomaly Detection in Surveillance Videos},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages     = {6479--6488},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00678}
}
```

### SO-TAD

```bibtex
@article{chen2025sotad,
  author  = {Chen, Xingyuan and Xu, Huahu and Ruan, Mingyang and Bian, Minjie and Chen, Qishen and Huang, Yuzhe},
  title   = {{SO-TAD}: A Surveillance-Oriented Benchmark for Traffic Accident Detection},
  journal = {Neurocomputing},
  volume  = {618},
  pages   = {129061},
  year    = {2025},
  doi     = {10.1016/j.neucom.2024.129061}
}
```

### CADP

```bibtex
@inproceedings{shah2018cadp,
  author    = {Shah, Ankit Parag and Lamare, Jean-Baptiste and Nguyen-Anh, Tuan and Hauptmann, Alexander},
  title     = {{CADP}: A Novel Dataset for CCTV Traffic Camera Based Accident Analysis},
  booktitle = {2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)},
  pages     = {1--9},
  year      = {2018},
  doi       = {10.1109/AVSS.2018.8639160}
}
```
