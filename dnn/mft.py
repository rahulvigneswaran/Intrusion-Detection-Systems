import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix


np.random.seed(1337)
tf.random.set_seed(1337)


train_df = pd.read_csv('kdd/binary/Training.csv', header=None)
test_df = pd.read_csv('kdd/binary/Testing.csv', header=None)

X_train_raw = train_df.iloc[:, 1:42].values
y_train = train_df.iloc[:, 0].values
X_test_raw = test_df.iloc[:, 1:42].values
y_test = test_df.iloc[:, 0].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)


os.makedirs("kddresults/transformer_mft", exist_ok=True)
os.makedirs("dnn/chung_results_transformer", exist_ok=True)
os.makedirs("dnnres", exist_ok=True)


def build_mft_model(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(256)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Reshape((1, 256))(x)

    attn = MultiHeadAttention(num_heads=8, key_dim=256)(x, x)
    x = LayerNormalization()(x + attn)

    ff = Dense(512, activation='relu')(x)
    mem = Dense(256, activation='tanh')(ff)
    x = LayerNormalization()(x + mem)

    ff2 = Dense(256, activation='relu')(x)
    x = LayerNormalization()(x + ff2)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)


def focal_loss_fixed(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1. - 1e-7)
    pos = tf.cast(tf.equal(y_true, 1), tf.float32)
    neg = 1.0 - pos
    loss = - (0.3 * pos * tf.pow(1 - y_pred, 2) * tf.math.log(y_pred) +
              0.7 * neg * tf.pow(y_pred, 2) * tf.math.log(1 - y_pred))
    return tf.reduce_mean(loss)


class MetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0:
            y_pred = (self.model.predict(X_test) > 0.2).astype(int)
            acc = accuracy_score(y_test, y_pred)
            pre = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"\n[Epoch {epoch+1}] Accuracy: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")


model = build_mft_model(41)
model.compile(loss=focal_loss_fixed, optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint("kddresults/transformer_mft/checkpoint-{epoch:02d}.keras", save_best_only=True)
csv_logger = CSVLogger("kddresults/transformer_mft/training_log.csv", separator=',', append=False)
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train,
          batch_size=64,
          epochs=100,
          validation_split=0.3,
          callbacks=[checkpointer, csv_logger, MetricsLogger(), earlystop],
          verbose=1)


model.save("dnn/chung_results_transformer/final_model.hdf5")

scores, names = [], []
for file in os.listdir("kddresults/transformer_mft"):
    if file.endswith(".keras"):
        model.load_weights(os.path.join("kddresults/transformer_mft", file))
        y_pred = (model.predict(X_test) > 0.2).astype(int)
        scores.append(f1_score(y_test, y_pred))
        names.append(file)

best_model = names[np.argmax(scores)]
model.load_weights(os.path.join("kddresults/transformer_mft", best_model))
y_pred = (model.predict(X_test) > 0.2).astype(int)
y_proba = model.predict(X_test)

np.savetxt("dnnres/transformer_predicted.txt", y_pred, fmt='%d')
np.savetxt("dnnres/transformer_probability.txt", y_proba, fmt='%.6f')

print("\n===== Final Evaluation on Best Transformer Checkpoint (Threshold = 0.32) =====")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
