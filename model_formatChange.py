# Load using older Keras
from keras import models
model = models.load_model("pancreatic_cancer_detection_model.h5")

# Save in the new format
model.save("model/pancreatic_cancer_model")

# Load using new format
model = models.load_model("model/pancreatic_cancer_model")
