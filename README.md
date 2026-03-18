# One-to-More: High-Fidelity Training-Free Anomaly Generation with Attention Control


## Installation

1. Create and activate a new Conda environment:

```shell
conda env create -f environment.yml
conda activate O2MAG
```



## Mask Generation

We use AnomalyDiffusion masks for MVTec-AD and SeaS for VisA/Real-IAD.

We have released the generated 500 image-mask pairs for MVTec-AD.

https://drive.google.com/drive/folders/1_RxRJy-PqFTEgja3vdkwf7BnaVhnPgrl?usp=drive_link

## Normal Data Augmentation

See Appendix C.1. for detailed information.

```
python ./img_augment.py
```



## Anomaly Generation

We provide three ways to run and evaluate our anomaly generation code:

1. **Interactive Web UI** (Requires approx. 14GB VRAM)

   Quickly edit and visualize anomalies in your browser.

```shell
python ./app_edit_anomaly_mask.py
```

------

2. **Jupyter Notebook** (Requires approx. 16GB VRAM)

```
edit_anomaly_mask.ipynb
```

3. Generate 1000 anomaly images per anomaly type.  About 30G

```
python edit_anomaly_moregpu_oneshot.py --root ./datasets/mvtec \
    --normal_path ./data_agument
    --sourece_image_mask ./anomalydiffusion/generated_mask \
    --embedding_file ./embed_bank/mvtec \
    --outputs_path ./generated_data_fewshot/mvtec \
    --pairs-file name-mvtec.txt --devices cuda:2,cuda:3,cuda:4,cuda:5
```

