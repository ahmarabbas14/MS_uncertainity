# Setup the Orion directory

```
$HOME
   ├── MS_uncertainity
   │   ├── config
   │   |   ├── 3d_unet.json
   │   |   └── (other configurations)
   |   ├── outputs
   │   |   └── README.md
   │   ├── customize_obj.py
   │   ├── experiment.py
   │   ├── slurm.sh
   │   └── (other files)
   ├── datasets (Put your datasets in here)
   │   └── headneck
   │       ├── headneck_3d_new.h5
   │       └── (other datasets)
   ├── hnperf (log files will be saved in here)
   │
```
```
$SCRATCH (/mnt/SCRATCH/ahab)
 ├── hnperf
 ├── analysis
```


# Run experiments
From your `$HOME` directory
```bash
cd MS_uncertainity
sbatch slurm.sh config/3d_unet.json 3d_unet 100 --model_checkpoint_period 5 --prediction_checkpoint_period 5
```
Which will save the trained model every 5 epochs and predict the validation set every 5 epoch
