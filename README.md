
# SCAD Dataset Reconstruction

SCAD is a curated binary accident detection benchmark consisting of 1,122 four-second video clips (561 Normal and 561 Accident), derived from four publicly available research datasets.

To respect the original dataset providers' rights, SCAD provides reconstruction instructions rather than redistributing the source videos.

after you clone our repo. 


1. CADP (our toughest) (bear with me) we are curreently in teh midst of requesting accesss to publicly distribute their daat (they allow academic use but is unclear whether redistribution is accepted (similar fpr all 4 datasets) Therefore get ready to be in a ride with. We promis you it will not take more than 30 minutes of your personal time (well as long as you are willing to let rclone run for 2 hours. Then again we spent months creating and validating so I hope you can help us contribute by iving you guys a nice accident dtecetion benchmark.

Original dataset (please read their instructions, however it is very difficult to download this dataset directly, I ahve spend a few weeks wrapping my head around this. so we have brough to a nice tehcnique to get thsi dataset)
Their github: https://ankitshah009.github.io/accident_forecasting_traffic_camera
And what we are after: https://drive.google.com/drive/folders/1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v

SCAD will requires 168. which is in our cadp_include.txt

**Step 1. Install Rclone** at https://rclone.org/downloads/

**Step 2. Configure Google Drive** find the folder with rrclone config


Create a Google Drive remote named `cadp` and authorize access using your Google account.

For large downloads, a personal Google Drive API client is recommended:
https://rclone.org/drive/#making-your-own-client-id

**Step 3. Download the required frame folders**

The provided `manifests/cadp_required_ids.txt` contains the 168 required source IDs.

Run the preparation script to generate the Rclone filter:

```bash
python scripts/prepare_cadp_filter.py
```

Then download:

```bash
rclone copy "cadp,root_folder_id=1BRIv9h4c_zbqJ6Ye4rVyFSo3K20BQt_v:" "CADP/" --include-from manifests/cadp_include.txt --transfers 8 --checkers 8 --tpslimit 4 -P
```

Only the required CADP folders will be downloaded. Interrupted downloads can be resumed by running the same command again.

**Note:** The CADP source-ID mapping and frame timing must be verified against the SCAD manifest before reconstruction.

### 2.2 SO-TAD

Original dataset:
https://github.com/cccxy-299/so-tad

Download the dataset from its official repository and place the files in `SO-TAD/`.

Detailed preparation instructions will be added after source verification.

### 2.3 TU-DAT

Download TU-DAT from the original authors' official distribution source and place it in `TU-DAT/`.

Detailed preparation instructions and the verified source link will be added.

### 2.4 UCF-Crime

Original dataset:
https://www.crcv.ucf.edu/projects/real-world/

Download the original dataset and place the videos in `UCF-Crime/`.

Detailed preparation instructions will be added after source verification.

## 3. Reconstruct SCAD

Once all four datasets have been downloaded, run:

```bash
python scripts/prepare_scad.py
```

The reconstruction pipeline will:

1. Identify the required source videos.
2. Extract 4-second clips with a 3-second stride.
3. Retain clips listed in the SCAD manifest.
4. Assign Normal/Accident labels.
5. Apply the predefined train, validation and test partitions.
6. Verify the reconstructed dataset.

The final output should contain:

```text
SCAD_output/
├── train/
├── val/
└── test/
```

Expected total: 1,122 clips.

The reconstruction script and complete manifests will be provided with the final release.

## 4. Dataset Usage

SCAD is intended for non-commercial academic and scientific research.

The original video content remains subject to the rights and usage conditions of CADP, SO-TAD, TU-DAT and UCF-Crime.

Researchers must obtain the original datasets through authorized sources and cite the relevant original publications alongside SCAD.

## 5. Citation

The official SCAD publication citation will be added upon release.
