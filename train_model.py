# USAGE
# python train_model.py

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

# Define paths for the embeddings, recognizer, and label encoder
EMBEDDINGS_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/output/PyPower_embed.pickle"
RECOGNIZER_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/output/PyPower_recognizer.pickle"
LABEL_ENCODER_PATH = "D:/PyPower_face-recognition-20240702T181609Z-001/PyPower_face-recognition/output/PyPower_label.pickle"

# load the face embeddings
print("[INFO] loading face embeddings...")
with open(EMBEDDINGS_PATH, "rb") as f:
    data = pickle.load(f)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
with open(RECOGNIZER_PATH, "wb") as f:
    f.write(pickle.dumps(recognizer))

# write the label encoder to disk
with open(LABEL_ENCODER_PATH, "wb") as f:
    f.write(pickle.dumps(le))

print("[INFO] model and label encoder saved.")
