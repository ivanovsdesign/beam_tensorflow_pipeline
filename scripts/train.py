import tensorflow as tf
import tensorflow_datasets as tfds

# Assume this is a function that returns a Keras model
from models.model import build_model


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def main():
    # Load and preprocess data
    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.map(preprocess).batch(32)
    test_dataset = test_dataset.map(preprocess).batch(32)

    # Build and train model
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=5)

    # Save the model
    model.save('models/saved_model/mnist_model')


if __name__ == '__main__':
    main()
