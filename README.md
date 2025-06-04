# Team SeRRa @ CheckThat! 2025

This projects contains the notebooks used by Team SeRRa for the 2025 edition of the CheckThat! Lab @ Clef 2025 submission.

## How-to install

It is recommended to install the required packages in it own virtual env

    python3 -m venv .env

Then install the packages listed in *requirements.txt* file

    pip3 install -r requirements.txt

## How-to train the models

Before running, it is necessary to have an valid account in the HuggingFaces hub, since the base models are downloaded from there.  It is also necessary to download the challenge dataset, which can be found at: https://gitlab.com/checkthat_lab/clef2025-checkthat-lab/-/tree/main/task4/subtask_4b.

After downloading the necessary data, start by training the binary classifier. To train the model, it is possible to use the notebook:

    data-prep/prepare-classifier-training-dataset.ipynb

to prepare the necessary data set, then the notebook:

    trainers/pairwise-trainer.ipynb

to execute the actual training process.


To train the pairwise compartor model, first prepare the training data for the model. Run the notebook with the *train* split of the dataset:

    preselect-hard-pairs.ipynb

This notebook autmatically runs the 1st and 2nd step of the pipeline, producing that will serve as the negative samples Then, to build the dataset, use the notebook:

    data-prep/prepare-pairwise-comparator-training-dataset.ipynb

and finally the training script in the notebook:

    trainers/pairwise-trainer.ipynb

At the end, there should be 2 fine-tuned models inside the *models* folder.

## How-to run

After training the models, running the pipeline is straigthforward, with a different notebook being responsible by each of the pipeline's step.

Start by running the script in

    bi-encoder-ranking.ipynb

Followed by

    classifier-ranking.ipynb

and finally

    pairwise-ranking.ipynb


At the end of each step, a partial prediction is produced, and can be found in the folder *parital-predictions*.

