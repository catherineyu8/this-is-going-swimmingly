# this-is-going-swimmingly

## data downloading/preprocessing

MMSD 2.0 dataset

-   To download and preprocess the MMSD 2.0 dataset, run data/load_data.py
-   This will generate a folder called "mmsd_processed" which contains MMSD 2.0 training/validation/testing data splits
-   Do NOT upload this to github! (it is in the .gitignore)

MUSE dataset

-

## notes to run the model

To train the model

-   src/main.py --mode test

To test the model on MMSD 2.0 dataset test split

-   src/main.py --mode train --dataset mmsd2.0

To test the model on the MUSE dataset (train split)

-   src/main.py --mode train --dataset muse

To run a container with tensorflow in oscar:

-   apptainer run --nv /oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg
