import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model


def build(
    time_steps: int,
    n_features:  int,
    units:       list[int] = [128, 64],
    dropout:     float = 0.2,
) -> Model:
    inp = Input(shape=(time_steps, n_features))
    x   = Bidirectional(LSTM(units[0], return_sequences=True))(inp)
    x   = Dropout(dropout)(x, training=True)
    x   = LSTM(units[1])(x)
    x   = Dropout(dropout)(x, training=True)
    out = Dense(1)(x)

    model = Model(inp, out, name="StockLSTM")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    model.summary()
    return model


def train(
    model:      Model,
    X_train:    np.ndarray,
    y_train:    np.ndarray,
    X_val:      np.ndarray,
    y_val:      np.ndarray,
    epochs:     int = 150,
    batch_size: int = 32,
    save_path:  str = "best_model.keras",
):
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )


def mc_predict(
    model:     Model,
    X:         np.ndarray,
    n_samples: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.stack(
        [model(X, training=True).numpy().flatten() for _ in range(n_samples)],
        axis=0,
    )
    return samples.mean(axis=0), samples.std(axis=0)


def forecast_future(
    model:       Model,
    last_window: np.ndarray,
    n_days:      int,
    n_samples:   int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    buf = list(last_window)
    ts  = len(last_window)
    all_preds = []

    for _ in range(n_days):
        x  = np.array(buf[-ts:]).reshape(1, ts, -1)
        mc = np.array([model(x, training=True).numpy()[0, 0] for _ in range(n_samples)])
        all_preds.append(mc)
        mean_pred = mc.mean()
        next_row  = buf[-1].copy()
        next_row[0] = mean_pred
        buf.append(next_row)

    arr = np.array(all_preds)
    return arr.mean(axis=1), arr.std(axis=1)
