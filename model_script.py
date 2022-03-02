import sys
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle
import argparse
from os import path


# Script to create and train basic linear SVC to predict website classifications
# based on this 14'000 entries data-set https://www.kaggle.com/hetulmehta/classification-of-websites

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default='/home/hypenguin/datasets/website_classification.csv',
                    help="path to website classification training set.")

args = parser.parse_args()

training_set_path = args.input

if not path.exists(training_set_path):
    print("Please specify a valid path to the training set...")
    sys.exit()

print("Loading training set from {}... ".format(training_set_path))

dataset = pd.read_csv(training_set_path)
df = dataset[['url', 'text', 'category']].copy()

df['category_id'] = df['category'].factorize()[0]
category_id_df = df[['category', 'category_id']].drop_duplicates()

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

X = df['text']
y = df['category']

labels = df.category_id

print("Creating model for output..")

X_train, X_test, y_train, y_test = train_test_split(X, df['category_id'],  test_size=0.25, random_state=0, shuffle=True)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)

model = LinearSVC()
model.fit(tfidf_vectorizer_vectors, y_train)
calibrated_classifier = CalibratedClassifierCV(base_estimator=model, cv="prefit")
calibrated_classifier.fit(tfidf_vectorizer_vectors, y_train)
m1 = calibrated_classifier

print("Writing model to binary files...")

# dump the binaries, use them separately in another script to do predictions

with open("FittedVectorizer.DAT", "wb") as f:
    f.write(pickle.dumps(fitted_vectorizer))


with open("CalibratedClassifierCV.DAT", "wb") as f:
    f.write(pickle.dumps(m1))

print("Done.")
