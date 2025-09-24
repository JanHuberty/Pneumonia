from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Flatten
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
def count_images_in_folder(dataset_base_dir):
    image_count = 0
    for class_folder in os.listdir(dataset_base_dir):
        class_folder_path = os.path.join(dataset_base_dir, class_folder)
        if os.path.isdir(class_folder_path): 
            num_images = len(os.listdir(class_folder_path))
            print(f"Number of images in '{class_folder}' folder: {num_images}")
            image_count += num_images
    return image_count

train_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/train/'
val_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/'
test_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'
print("Training Set:")
train_images_count = count_images_in_folder(train_dir)
print(f"Total images in training set: {train_images_count}\n")
print("Validation Set:")
val_images_count = count_images_in_folder(val_dir)
print(f"Total images in validation set: {val_images_count}\n")
print("Test Set:")
test_images_count = count_images_in_folder(test_dir)
print(f"Total images in test set: {test_images_count}\n")
class_names = ['Normal', 'Pneumonia']
val_images, val_labels = next(val_generator)
normal_images = val_images[val_labels == 0]
pneumonia_images = val_images[val_labels == 1]
num_images_to_display = min(5, len(normal_images), len(pneumonia_images))
fig, axes = plt.subplots(2, num_images_to_display, figsize=(15, 6), squeeze=False)  
for i in range(num_images_to_display):
    ax = axes[0, i] 
    ax.imshow(normal_images[i], cmap='gray')  
    ax.set_title(class_names[0])  
    ax.axis('off')
for i in range(num_images_to_display):
    ax = axes[1, i]  
    ax.imshow(pneumonia_images[i], cmap='gray')  
    ax.set_title(class_names[1])  
    ax.axis('off')
plt.tight_layout()
plt.show()
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,  
    horizontal_flip=True,
    rotation_range=30,  
    width_shift_range=0.3,  
    height_shift_range=0.2,  
    brightness_range=[0.8, 1.2],  
    channel_shift_range=0.2,
    validation_split=0.2 )

batch_size = 16
image_size = (224, 224)

train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=image_size,  
    batch_size=batch_size,
    class_mode='binary',  
    subset='training')
val_generator = train_datagen.flow_from_directory(
    train_dir,  
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  )
test_datagen = ImageDataGenerator(rescale=1./255)  
test_generator = test_datagen.flow_from_directory(
    test_dir,  
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  )
print("Class indices for training:", train_generator.class_indices)
print("Class indices for validation:", val_generator.class_indices)
labels = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:50]:
    layer.trainable = False
for layer in base_model.layers[50:]:
    layer.trainable = True
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  
x = Dropout(0.5)(x) 
x = BatchNormalization()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  
model = Model(inputs=base_model.input, outputs=predictions)
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)
lr_scheduler = LearningRateScheduler(scheduler)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=1e-6, verbose=1)
model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss='binary_crossentropy',  
    metrics=['accuracy'])  
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,  
    class_weight=class_weights,  
    callbacks=[lr_scheduler, early_stopping, lr_reduction])
model.save('/kaggle/working/final_model.h5')
for layer in base_model.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=SGD(learning_rate=1e-5, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[lr_reduction, early_stopping],  
    class_weight=class_weights)
model.save('/kaggle/working/fine_tune_final_model.h5')
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
y_true = test_generator.classes 
y_pred_prob = model.predict(test_generator)  
y_pred = np.where(y_pred_prob > 0.5, 1, 0)  

print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")

conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues,
            xticklabels=["Healthy", "Pneumonia"], yticklabels=["Healthy", "Pneumonia"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_pred_prob)  
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
