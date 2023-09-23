#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[2]:


pip install numpy


# In[3]:


pip install tensorflow


# In[16]:


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# URL to the MobileNetV2 model on TensorFlow Hub
mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

# Load the MobileNetV2 model
mobilenet_model = hub.load(mobilenet_url)

# Load an example image for classification
image_path = "testing.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB color space
image = cv2.resize(image, (224, 224))  # Resize to match the input size of MobileNetV2
image = image / 255.0  # Normalize pixel values to [0, 1]

# Make predictions with the model
predictions = mobilenet_model(image[np.newaxis, ...])  # Add batch dimension

# Decode the predictions
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())

# Display the top predicted classes
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"Prediction {i + 1}: {label} ({score:.2f})")

# Display the image
cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[17]:


import tensorflow as tf

# Ensure the image is a TensorFlow tensor
image = tf.convert_to_tensor(image, dtype=tf.float32)

# Resize and normalize the image
image = tf.image.resize(image, (224, 224))
image = image / 255.0

# Add a batch dimension to match the model's input shape
image = tf.expand_dims(image, axis=0)

# Make predictions with the model
predictions = mobilenet_model(image)

# Decode the predictions
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())

# Display the top predicted labels and their confidence scores
for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
    print(f"Prediction {i + 1}: {label} ({score:.2f})")


# In[18]:


import numpy as np

# Assuming predictions is a 2D array of shape (1, num_classes)
# Replace `num_classes` with the actual number of classes in your model
num_classes = 1000  # Adjust this value based on your model

# Get the top predicted class index
top_class_index = np.argmax(predictions, axis=1)[0]

# Replace this with your own class labels or mapping
class_labels = ["Class1", "Class2", "Class3", ...]  # Replace with your class labels

# Get the predicted class label
predicted_class = class_labels[top_class_index]

# Get the confidence score for the predicted class
confidence_score = predictions[0, top_class_index]

print(f"Predicted class: {predicted_class}")
print(f"Confidence score: {confidence_score:.2f}")


# In[19]:


import numpy as np

# Assuming predictions is a 2D array of shape (1, num_classes)
# Replace `num_classes` with the actual number of classes in your model
num_classes = 1000  # Adjust this value based on your model

# Print the shape of the predictions array to understand its structure
print(predictions.shape)

# Get the top predicted class index
top_class_index = np.argmax(predictions, axis=1)[0]

# Print the top class index for inspection
print(f"Top Class Index: {top_class_index}")


# In[20]:


import cv2

# Load your image (replace with your image loading code)
image = cv2.imread("example_image.jpg")

# Perform mask detection and obtain bounding boxes (replace with your detection code)
# For example, you might have code like this:
# detected_objects, bounding_boxes = perform_mask_detection(image)

# Loop through detected objects and draw bounding boxes
for bbox in bounding_boxes:
    x, y, w, h = bbox  # Get the coordinates and dimensions of the bounding box
    color = (0, 255, 0)  # Green color (you can change this)
    thickness = 2  # Line thickness (you can change this)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

# Save or display the image with bounding boxes
cv2.imwrite("detected_image.jpg", image)  # Save the image with bounding boxes
cv2.imshow("Detected Image", image)  # Display the image with bounding boxes
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[21]:


import cv2

# Load your image (replace with your image loading code)
image = cv2.imread("testing.jpg")

# Perform mask detection and obtain bounding boxes (replace with your detection code)
# For example, you might have code like this:
# detected_objects, bounding_boxes = perform_mask_detection(image)

# Loop through detected objects and draw bounding boxes
# Replace the comments below with your actual detection code
# for bbox in bounding_boxes:
#     x, y, w, h = bbox  # Get the coordinates and dimensions of the bounding box
#     color = (0, 255, 0)  # Green color (you can change this)
#     thickness = 2  # Line thickness (you can change this)
#     cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

# Save or display the image with bounding boxes
# cv2.imwrite("detected_image.jpg", image)  # Save the image with bounding boxes
# cv2.imshow("Detected Image", image)  # Display the image with bounding boxes
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# In[22]:


import cv2

# Load your image (replace with your image loading code)
image = cv2.imread("testing.jpg")

# Perform mask detection and obtain bounding boxes (replace with your detection code)
# For example, you might have code like this:
# detected_objects, bounding_boxes = perform_mask_detection(image)

# Loop through detected objects and draw bounding boxes
# Replace the comments below with your actual detection code
# for bbox in bounding_boxes:
#     x, y, w, h = bbox  # Get the coordinates and dimensions of the bounding box
#     color = (0, 255, 0)  # Green color (you can change this)
#     thickness = 2  # Line thickness (you can change this)
#     cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

# Display the image with bounding boxes
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


import cv2
import matplotlib.pyplot as plt

# Load your image (replace with your image loading code)
image = cv2.imread("example_image.jpg")

# Perform mask detection and obtain bounding boxes (replace with your detection code)
# For example, you might have code like this:
# detected_objects, bounding_boxes = perform_mask_detection(image)

# Loop through detected objects and draw bounding boxes
# Replace the comments below with your actual detection code
# for bbox in bounding_boxes:
#     x, y, w, h = bbox  # Get the coordinates and dimensions of the bounding box
#     color = (0, 255, 0)  # Green color (you can change this)
#     thickness = 2  # Line thickness (you can change this)
#     cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

# Display the image with bounding boxes using Matplotlib
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis labels
plt.show()


# In[ ]:




