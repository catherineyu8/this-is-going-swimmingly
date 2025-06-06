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

FLICKR dataset download & preprocessing

-   Run data/load_data_flickr.py
-   This will generate a folder called "flickr_processed" which contains ~3000 random data points from flickr
-   Do NOT upload this to github! (it is in the .gitignore)

SarcNet preprocessing
-   Run data/load_data_sarcnet.py (no need to have the data locally; it's taken from a google drive link)
-   This will generate a folder called "sarcnet_processed" which contains SarcNet train/test/val data split
-   That folder is also in the .gitignore

## notes to run the model

To train the model

-   src/main.py --mode train

To test the model on MMSD 2.0 dataset test split

-   src/main.py --mode test --dataset mmsd2.0

To test the model on the MUSE/FLICKR custom combined dataset

-   src/main.py --mode test --dataset muse_flickr

To test the model on our own data

-   src/main.py --mode test --dataset us

To test the model on SarcNet data

-   src/main.py --mode test --dataset sarcnet

To run a container with tensorflow in oscar:

-   apptainer run --nv /oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg
