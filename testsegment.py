import os
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm



# Set your paths
JSON_DIR = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\jsonfiles'
TRAIN_MASK_PATH = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\train\masks'
TRAIN_IMAGE_PATH = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\train\images'
TEST_IMAGE_PATH = r'C:\Users\kelvi\OneDrive\Desktop\woodcutter2\segmentation\test\images'

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
NUM_CLASSES = 3

# Load training images and masks
# Load training images and masks
# Load training images and masks
train_ids = next(os.walk(TRAIN_IMAGE_PATH))[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=bool)

print('Resizing training images and masks')
for n, file in tqdm(enumerate(train_ids), total=len(train_ids)):
    image_path = os.path.join(TRAIN_IMAGE_PATH, file)
    mask_file_name = os.path.splitext(file)[0] + '.png'
    mask_path = os.path.join(TRAIN_MASK_PATH, mask_file_name)

    # Load image
    img = imread(image_path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img

    # Load mask
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES), dtype=bool)
    mask_ = imread(mask_path)
    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    for class_idx in range(NUM_CLASSES):
        mask[:, :, class_idx] = mask_ * (mask_ == class_idx)
    Y_train[n] = mask

    # Count occurrences of each class label
    unique_labels, label_counts = np.unique(mask_, return_counts=True)

# Print the counts for each class label
    for label, count in zip(unique_labels, label_counts):
        print(f'{count} occurrences of label {label}')

# Example of mapping the counts to your model for training
    tree_label_count = label_counts[1]
    ground_label_count = label_counts[0]
    sky_label_count = label_counts[2]

# Load test images
test_ids = next(os.walk(TEST_IMAGE_PATH))[2]

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []

print('Resizing test images')
for n, file in tqdm(enumerate(test_ids), total=len(test_ids)):
    image_path = os.path.join(TEST_IMAGE_PATH, file)

    # Load image
    img = imread(image_path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img


#BUILD THE MODEL
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
#Contraction Path

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.2)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='softmax')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Model checkpoints
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='c:/Users/kelvi/OneDrive/Desktop/woodcutter2/segmentation.h5', verbose=1, save_best_only=True)
#model.save('segmentation.h5')
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    checkpointer
]

#model = tf.keras.models.load_model('segmentation.h5')
# Train the model
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=15, callbacks=callbacks)

# Load the new image
new_image_path = 'C:/Users/kelvi/OneDrive/Desktop/woodcutter2/test_images/SYCW0398.jpg'
new_image = imread(new_image_path)
new_image = resize(new_image, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
new_image = new_image / 255.0  # Normalize the image
new_image = np.expand_dims(new_image, axis=0)

# Make predictions on the new image
prediction = model.predict(new_image)
predicted_mask = np.argmax(prediction, axis=-1)
print (prediction.shape)
print (predicted_mask.shape)
import matplotlib.pyplot as plt

# Map the predicted mask to RGB color values
color_map = {
    0: [255, 0, 0],    # Class 0 - Ground (red)
    1: [0, 255, 0],    # Class 1 - Tree (green)
    2: [0, 0, 255],    # Class 2 - Sky (blue)
}

predicted_mask_rgb = np.zeros((predicted_mask.shape[1], predicted_mask.shape[2], 3), dtype=np.uint8)
for class_idx, color in color_map.items():
    color = np.array(color, dtype=np.uint8)
    predicted_mask_rgb[predicted_mask[0, :, :] == class_idx] = color
##########################    
print("predicted_mask:")
print(predicted_mask)

print("predicted_mask_rgb:")
print(predicted_mask_rgb)

print("color_map:")
print(color_map)
###########################

# Print the unique class indices in the predicted mask
unique_class_indices = np.unique(predicted_mask[..., 0])
print("Unique class indices in predicted_mask:")
print(unique_class_indices)
# Plot the original image and the predicted mask
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(new_image[0])
axes[0].set_title('Original Image')
axes[0].axis('off')
axes[1].imshow(predicted_mask_rgb)
axes[1].set_title('Predicted Mask')
axes[1].axis('off')

plt.show()