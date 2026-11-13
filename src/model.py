"""
src/model.py — the Convolutional Neural Network (CNN).

Architecture (input 64x64x3 RGB, values 0..1):

    Block 1:  Conv(32) -> BN -> ReLU -> Conv(32) -> BN -> ReLU -> MaxPool -> Dropout   64x64 -> 32x32
    Block 2:  Conv(64) -> BN -> ReLU -> Conv(64) -> BN -> ReLU -> MaxPool -> Dropout   32x32 -> 16x16
    Block 3:  Conv(128)-> BN -> ReLU -> Conv(128)-> BN -> ReLU -> MaxPool -> Dropout   16x16 ->  8x8
    Head:     GlobalAveragePooling -> Dense(128, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)

Why this design?
  * Parasites are small purple-stained dots/rings inside a cell. 3x3 convolutions
    are good at detecting such small local patterns; stacking blocks lets later
    layers combine edges -> textures -> "parasite-like" shapes.
  * Filters double (32 -> 64 -> 128) while the image shrinks, the classic
    VGG-style trade-off between detail and abstraction.
  * ~0.3 million parameters: big enough for this task (published custom CNNs on
    this dataset are similar), small enough to train on a laptop CPU.
  * Regularisation against overfitting: L2 weight decay, Batch Normalisation,
    Dropout, plus data augmentation in the input pipeline.
  * One sigmoid output = probability that the cell is Parasitized.
    P(Uninfected) = 1 - P(Parasitized).

Transfer learning (e.g. MobileNet/ResNet pretrained on ImageNet) was considered,
but those models expect larger inputs (96-224 px), are 10-80x larger and much
slower on a CPU and for LIME/SHAP, while small custom CNNs already perform well
on these simple single-cell images.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def build_cnn(image_size: int = config.IMAGE_SIZE,
              l2_weight: float = config.L2_WEIGHT,
              dropout_conv: float = config.DROPOUT_CONV,
              dropout_dense: float = config.DROPOUT_DENSE):
    """Create (but don't train) the CNN."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers

    reg = regularizers.l2(l2_weight)

    def conv_block(x, filters, block):
        for i in (1, 2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_regularizer=reg, name=f"block{block}_conv{i}")(x)
            x = layers.BatchNormalization(name=f"block{block}_bn{i}")(x)
            x = layers.Activation("relu", name=f"block{block}_relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"block{block}_pool")(x)
        return layers.Dropout(dropout_conv, name=f"block{block}_dropout")(x)

    inputs = layers.Input(shape=(image_size, image_size, config.CHANNELS), name="cell_image")
    x = conv_block(inputs, 32, 1)
    x = conv_block(x, 64, 2)
    x = conv_block(x, 128, 3)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="dense_128")(x)
    x = layers.Dropout(dropout_dense, name="dense_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_parasitized")(x)
    return models.Model(inputs, outputs, name="malaria_cnn")


def compile_model(model, learning_rate: float = config.LEARNING_RATE):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def load_trained_model(path: Path = config.MODEL_PATH):
    """Load the saved model (no retraining). compile=False: we only need predictions."""
    import tensorflow as tf

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Train it first with:  python -m src.train")
    return tf.keras.models.load_model(str(path), compile=False)


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
