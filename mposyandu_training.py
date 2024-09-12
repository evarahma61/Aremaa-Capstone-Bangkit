# This file is for training on AI Platform with scikit-learn.


# [START setup]
import datetime
import os
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn import svm
import joblib

# Fill in your Cloud Storage bucket name
BUCKET_NAME = 'x-circle-314022-mposyandu'
# [END setup]

# [START download-data]
mposyandu_data_filename = 'mposyandu_data.csv'
mposyandu_target_filename = 'mposyandu_target.csv'
data_dir = 'gs://x-circle-314022-mposyandu'

# gsutil outputs everything to stderr so we need to divert it to stdout.
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    mposyandu_data_filename),
                       mposyandu_data_filename], stderr=sys.stdout)
subprocess.check_call(['gsutil', 'cp', os.path.join(data_dir,
                                                    mposyandu_target_filename),
                       mposyandu_target_filename], stderr=sys.stdout)
# [END download-data]


# [START load-into-pandas]
# Load data into pandas, then use `.values` to get NumPy arrays
mposyandu_data = pd.read_csv(mposyandu_data_filename).values
mposyandu_target = pd.read_csv(mposyandu_target_filename).values

# Convert one-column 2D array into 1D array for use with scikit-learn
mposyandu_target = mposyandu_target.reshape((mposyandu_target.size,))
# [END load-into-pandas]

# [START train-and-save-model]
# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(mposyandu_data, mposyandu_target)

# Export the classifier to a file
model_filename = 'model.joblib'
joblib.dump(classifier, model_filename)
# [END train-and-save-model]


# [START upload-model]
# Upload the saved model file to Cloud Storage
gcs_model_path = os.path.join('gs://', BUCKET_NAME,
    datetime.datetime.now().strftime('iris_%Y%m%d_%H%M%S'), model_filename)
subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path],
    stderr=sys.stdout)
# [END upload-model]
