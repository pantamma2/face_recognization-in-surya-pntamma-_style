import os
import json
import cv2
import numpy as np
import tensorflow as tf
import albumentations as alb
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, GlobalMaxPooling2D, Dropout # type: ignore
from matplotlib import pyplot as plt

# Constants
IMAGE_SIZE = (120, 120)

# Data Augmentation
augmentor = alb.Compose([
    alb.RandomCrop(width=450, height=450),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

def get_user_input():
    image_dir = input("C:\\Users\\chsur\\OneDrive\\Desktop\\suri-all\\data2\\images")
    labels_dir = input("C:\\Users\\chsur\\OneDrive\\Desktop\\suri-all\\data2\\labels")
    return image_dir, labels_dir

def load_image(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    return img

def process_image_and_labels(image_path, label_path):
    img = cv2.imread(image_path)
    coords = [0, 0, 0.00001, 0.00001]
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            label = json.load(f)
        coords[0] = label['shapes'][0]['points'][0][0]
        coords[1] = label['shapes'][0]['points'][0][1]
        coords[2] = label['shapes'][0]['points'][1][0]
        coords[3] = label['shapes'][0]['points'][1][1]
        coords = list(np.divide(coords, [640, 480, 640, 480]))
    return img, coords

def augment_image(image, coords):
    augmented = augmentor(image=image, bboxes=[coords], class_labels=['face'])
    aug_image = augmented['image']
    if len(augmented['bboxes']) == 0:
        aug_coords = [0, 0, 0, 0]
        aug_class = 0
    else:
        aug_coords = augmented['bboxes'][0]
        aug_class = 1
    return aug_image, aug_coords, aug_class

def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding="utf-8") as f:
        label = json.load(f)
    return [label['class']], label['bbox']

def preprocess_dataset(images, labels, augment_times=60):
    def generator():
        for img_path, lbl_path in zip(images, labels):
            img, coords = process_image_and_labels(img_path, lbl_path)
            for _ in range(augment_times):
                aug_img, aug_coords, aug_class = augment_image(img, coords)
                yield aug_img, (aug_class, aug_coords)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=(
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        (tf.TensorSpec(shape=(), dtype=tf.uint8), tf.TensorSpec(shape=(4,), dtype=tf.float32))
    ))
    dataset = dataset.map(lambda x, y: (tf.image.resize(x, IMAGE_SIZE) / 255.0, y))
    dataset = dataset.batch(8).prefetch(4)
    return dataset

def build_custom_model():
    input_layer = Input(shape=(120, 120, 3))
    
    # Custom convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Classification Model  
    f1 = Dense(256, activation='relu')(x)
    class2 = Dense(1, activation='sigmoid')(f1)
    
    # Bounding box model
    f2 = Dense(256, activation='relu')(x)
    regress2 = Dense(4, activation='sigmoid')(f2)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker

def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]
    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    return delta_coord + delta_size

class FaceTracker(Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs):
        X, y = batch
        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}
    
    def test_step(self, batch, **kwargs):
        X, y = batch
        classes, coords = self.model(X, training=False)
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss
        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}
        
    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

# Get user inputs
image_dir, labels_dir = get_user_input()

# Load original image and label paths
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

label_files = [os.path.join(labels_dir, f) for f in os.listdir(labels_dir) if f.endswith('*.json')]

# Split into training, validation, and testing sets (70%, 15%, 15%)
train_images, temp_images, train_labels, temp_labels = train_test_split(image_files, label_files, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5, random_state=42)

# Preprocess datasets
train = preprocess_dataset(train_images, train_labels)
val = preprocess_dataset(val_images, val_labels)
test = preprocess_dataset(test_images, test_labels)

# Build and compile model
facetracker = build_custom_model()
model = FaceTracker(facetracker)
batches_per_epoch = len(train)
lr_decay = (1. / 0.75 - 1) / batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)
model.compile(opt, tf.keras.losses.BinaryCrossentropy(), localization_loss)

# Train the model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

# Save and reload the model
model.save('facetracker.h5')
facetracker = tf.keras.models.load_model('facetracker.h5', custom_objects={'localization_loss': localization_loss})
