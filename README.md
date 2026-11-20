
# SCAD: Surveillance Curated Accident Dataset

**A curated benchmark for surveillance-based accident recognition.**

SCAD combines footage from **CADP, SO-TAD, TU-DAT, and UCF-Crime** for binary **normal vs. accident/anomalous** video classification. It contains **1,122 balanced four-second clips** (561 per class), with fixed **train/validation/test splits of 778/112/232**. All clips are **30 FPS (120 frames)**, supporting reproducible comparisons across video recognition models.

## Research Papers

**Dataset curation and initial benchmarking** (ETASR, June 2026)  
[From Chaos to Detection: Accident Benchmarking in Surveillance Videos with a Curated Dataset and 3D CNNs](https://doi.org/10.48084/etasr.17024)

**Follow-up video encoder benchmarking study** (In review; target release: early 2027)  
*A Comparative Study of Pretrained Video Encoders for Surveillance Accident Recognition*  
Paper link: **Coming soon**.

### Experimental Scripts for the Follow-up Study

The experimental scripts accompanying the video encoder benchmarking study are available in [`scripts/encoder_benchmark/`](scripts/encoder_benchmark/).

The scripts are organized according to the study's experimental framework:

- **Stage I: Model Screening** ([`stage1_screening/`](scripts/encoder_benchmark/stage1_screening/)): Initial screening of 37 pretrained configurations across 20 model groups.
- **Stage II: Latency Profiling** ([`stage2_profiling/`](scripts/encoder_benchmark/stage2_profiling/)): OpenCV and NVIDIA NVDEC file-to-decision latency profiling of 17 shortlisted configurations.
- **Stage III: Five-Seed Evaluation** ([`stage3_five_seed/`](scripts/encoder_benchmark/stage3_five_seed/)): Training, evaluation, and latency assessment of ten finalist configurations across five random seeds.
- **External Evaluation** ([`external_evaluation/`](scripts/encoder_benchmark/external_evaluation/)): Complementary experiments involving DAD, CCD, and ACCIDENT-derived surveillance data.
- **Analysis** ([`analysis/`](scripts/encoder_benchmark/analysis/)): Supporting experimental analysis and evidence extraction.

These scripts are provided as **research reference implementations and supporting experimental records** to document the methods employed in the study. They are preserved for methodological transparency and reproducibility reference rather than as a standalone executable package. Execution may require the original experimental environment, datasets, pretrained checkpoints, external model repositories, and associated dependencies.

## Reconstruct SCAD

To respect the original dataset creators and their distribution terms, SCAD provides reconstruction code and manifests rather than redistributing video footage. Researchers can obtain the four source datasets directly from their respective providers and follow the steps below to reproduce SCAD with its original labels and splits.

Some downloads may take time depending on your connection, but we've kept the process straightforward and reproducible.

**Requirements:** [Python 3.9+](https://www.python.org/downloads/), [FFmpeg and ffprobe](https://ffmpeg.org/download.html), [Git](https://git-scm.com/), and [7-Zip](https://www.7-zip.org/). Ensure `python`, `ffmpeg`, and `ffprobe` are available from your terminal.

### Clone the Repository

```powershell
git clone https://github.com/WhoCares258/SCAD.git
cd SCAD
```

**All four source folders go inside the cloned `SCAD` folder.** Run the commands below from that folder.

### 1. TU-DAT

1. Read the **dataset-access instructions below the abstract** in the [TU-DAT paper](https://www.mdpi.com/1424-8220/25/11/3259), then open the [Google Drive dataset folder](https://drive.google.com/drive/folders/1qsORgV4p4yn-hxsVFYKdRZePYL2m_ncj).
2. Click the **three-dot menu** beside `Positive_Vidoes` and choose **Download**. Google Drive may take several minutes to prepare the ZIP.
3. Extract the archive and place the `Positive_Vidoes` folder inside `SCAD/`, keeping its original spelling. Videos should appear as `Positive_Vidoes/*.mov`.

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

**Duplicate note:** During the initial SCAD curation, `47_2.mp4` and `106_2.mp4` were identified as duplicates, and `47_2.mp4` was removed while `106_2.mp4` was retained. However, the available SO-TAD release subsequently removed source `106` as a duplicate of source `47`. Therefore, reconstruction uses source `47`, segment 2 (`47_2`), as a substitute for the retained `106_2.mp4` entry, preserving its original filename, label, and split. `47_1.mp4` remains unchanged.

### 4. CADP

Project and access instructions: [CADP](https://ankitshah009.github.io/accident_forecasting_traffic_camera) | [Original JPEG frame collection](https://drive.google.com/drive/folders/1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v)

CADP provides numbered JPEG image sequences. The included **`manifests/cadp_include.txt`** selects only the **168 source folders** needed by SCAD.

1. Read the CADP project's access instructions and install [rclone](https://rclone.org/downloads/).
2. To speed up downloads and reduce shared-client rate limits, [create your own Google API client ID and secret](https://rclone.org/drive/#making-your-own-client-id) and use them when configuring rclone. Keep these credentials private.
3. Run `rclone config`. Create a Google Drive remote named **`cadp`**, enter your personal API credentials, and complete browser authorization. Under advanced settings, set **`root_folder_id`** to `1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v`. See the [rclone Google Drive guide](https://rclone.org/drive/) if prompted differently.
4. From the cloned `SCAD` folder, verify the include file and download the selected folders:

   ```powershell
   Test-Path .\manifests\cadp_include.txt
   rclone copy cadp: .\CADP --include-from .\manifests\cadp_include.txt -P
   ```

   `Test-Path` should show `True`. Frames will appear as `CADP/000002/0.jpg`, etc. A personal API client does not bypass Google Drive access restrictions or the source owner's permissions.

## Final Folder Layout

Once downloaded, your cloned repository should look like:

```text
SCAD/
├── README.md
├── scripts/
│   ├── reconstruct_scad.py
│   └── encoder_benchmark/
│       ├── stage1_screening/
│       ├── stage2_profiling/
│       ├── stage3_five_seed/
│       ├── external_evaluation/
│       └── analysis/
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

## Generate the Dataset

From inside `SCAD`, run:

```powershell
python .\scripts\reconstruct_scad.py
```

To regenerate existing clips:

```powershell
python .\scripts\reconstruct_scad.py --overwrite
```

The script reconstructs all **1,122 SCAD clips**, preserving their original filenames, labels, and dataset splits.

All clips have been successfully reconstructed and verified through SSIM comparison or manual visual inspection. The public reconstruction script was also tested against the original SCAD reference, reproducing the same validation results.

**SCAD is ready for reproducible research and benchmarking.**

## Use and Citation

**SCAD is intended for academic and non-commercial research. Redistribution of the original source footage or reconstructed SCAD video clips is not authorized by this repository.**

To respect the intellectual property and contributions of the original dataset creators, this repository distributes only reconstruction code, manifests, and documentation. Researchers must obtain the footage directly from its respective providers, comply with their access and usage conditions, and acknowledge the original sources.

All rights to the underlying videos remain with their respective rights holders. SCAD's research-use conditions do not replace or override the terms of the contributing datasets. Researchers considering other forms of use should obtain the appropriate permissions from the relevant rights holders.

We sincerely appreciate the researchers and institutions behind CADP, SO-TAD, TU-DAT, and UCF-Crime for making their valuable datasets available to the research community.

**If SCAD supports your research, please cite the SCAD dataset paper and the four contributing datasets below.**

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
