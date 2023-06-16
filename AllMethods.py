import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import f1_score


def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def create_vgg_model():
    model = tf.keras.models.Sequential([
        tf.keras.applications.VGG16(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    for layer in model.layers[:-4]:
        layer.trainable = False

    return model

def load_cnn_model():
    return tf.keras.models.load_model('cnnModel.h5', custom_objects={'f1': f1})

def load_vgg_model():
    return tf.keras.models.load_model('vggModel.h5', custom_objects={'f1': f1})


def train_model(model, train_generator, val_generator, model_name):
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['binary_accuracy', 'categorical_accuracy', f1])

    callbacks = [
        ModelCheckpoint(filepath='models/' + f'{model_name}.h5', monitor='val_loss', save_best_only=True),
        CSVLogger('models/' + f'{model_name}.csv'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
        EarlyStopping(monitor='val_loss', patience=3)
    ]

    history = model.fit(train_generator,
                        epochs=EPOCHS,
                        validation_data=val_generator,
                        callbacks=callbacks)

    return history


def evaluate_model(model, generator):
    loss, acc = model.evaluate(generator)
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")


def save_confusion_matrix(model, generator, model_name):
    val_preds = model.predict(generator)

    val_preds = np.argmax(val_preds, axis=1)

    val_labels = generator.classes

    cm = confusion_matrix(val_labels, val_preds)
    print('Confusion matrix:')
    print(cm)

    class_names = ['COVID-19', 'NORMAL', 'PNEUMONIA-BACTERIAL', 'PNEUMONIA-VIRAL']

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    fig, ax = plt.subplots(figsize=(7, 5)) # set the figure size to 10 inches wide and 6 inches tall

    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title('Confusion Matrix')
    plt.savefig(f'{model_name}.png') # use bbox_inches='tight' to prevent the labels from being cut off
    plt.show()


# Define the parameters for the model
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4

# Create the data generators
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('E:/datasets/second/',
                                                    target_size=IMG_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

val_generator = val_datagen.flow_from_directory('E:/datasets/second/',
                                                target_size=IMG_SIZE,
                                                batch_size=BATCH_SIZE,
                                                class_mode='categorical', subset='validation')

test_generator = test_datagen.flow_from_directory(
    'E:/datasets/second/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical', subset='validation')

cnn_model = load_cnn_model()
cnn_history = train_model(cnn_model, train_generator, val_generator, 'cnnModel')

evaluate_model(cnn_model, val_generator)

save_confusion_matrix(cnn_model, test_generator, 'cnn')

vgg_model = load_vgg_model()
vgg_history = train_model(vgg_model, train_generator, val_generator, 'vggModel')

evaluate_model(vgg_model, val_generator)

save_confusion_matrix(vgg_model, test_generator, 'vgg')