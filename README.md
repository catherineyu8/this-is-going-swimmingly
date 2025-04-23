# this-is-going-swimmingly

## data downloading/preprocessing

MMSD 2.0 dataset download & preprocessing

-   Run data/load_data.py
-   This will generate a folder called "mmsd_processed" which contains MMSD 2.0 training/validation/testing data splits
-   Do NOT upload this to github! (it is in the .gitignore)

MUSE dataset download & preprocessing

-   Create a folder called "muse" within the data folder
-   From https://github.com/LCS2-IIITD/Multimodal-Sarcasm-Explanation-MuSE, download the images (link found in the README) as well as the train_df.tsv, val_df.tsv, and test_df.tsv files. Put these all in the "muse" folder. It should contain the 3 tsv files and a folder named "images" containing all the images.
-   Run data/load_data_muse.py
-   This will generate a folder called "muse_processed" which contains MUSE training/validation/testing data splits
-   Do NOT upload the muse_processed or muse folders to github! (it is in the .gitignore)

## notes to run the model

To train the model

-   src/main.py --mode test

To test the model on MMSD 2.0 dataset test split

-   src/main.py --mode train --dataset mmsd2.0

To test the model on the MUSE dataset (train split)

-   src/main.py --mode train --dataset muse

To run a container with tensorflow in oscar:

-   apptainer run --nv /oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg
