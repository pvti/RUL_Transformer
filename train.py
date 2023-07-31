import os
import argparse
import datetime
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from utils.loss import MyHuberLoss
from utils.logger import get_logger

from tabtransformertf.models.fttransformer import (
    FTTransformerEncoder,
    FTTransformer,
    Time2Vec,
)
from tabtransformertf.utils.preprocessing import df_to_dataset


def parse_args():
    parser = argparse.ArgumentParser("RUL detection via FTTransformer training")

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/b1_2_hor_data_raw.npy",
        help="path to dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="path for saving model and log file",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--gpu", type=str, default="0", help="Select gpu to use")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "-cpr", "--compress_rate", type=str, default="[0.]*100", help="compress rate"
    )

    return parser.parse_args()


def create_model(time_features, numerical_features, numerical_data, optimizer):
    t2v_encoder = Time2Vec(time_features=time_features, kernel_size=16, t2v_emb_dim=16)
    ft_linear_encoder = FTTransformerEncoder(
        numerical_features=numerical_features,
        categorical_features=[],
        numerical_data=numerical_data,
        categorical_data=None,
        y=None,
        numerical_embedding_type="linear",
        embedding_dim=128,
        depth=4,
        heads=8,
        attn_dropout=0.2,
        ff_dropout=0.2,
        explainable=True,
    )
    # Pass th encoder to the model
    model = FTTransformer(
        encoder=ft_linear_encoder,
        t2v_encoder=t2v_encoder,
        out_dim=1,
        out_activation="relu",
    )

    model.compile(
        optimizer=optimizer,
        loss={"output": MyHuberLoss(threshold=0.05), "importances": None},
        metrics={
            "output": [tf.keras.metrics.RootMeanSquaredError(name="rmse")],
            "importances": None,
        },
    )

    return model


def main():
    args = parse_args()
    # logger
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger = get_logger(os.path.join(args.output_dir, now + ".txt"))
    random.seed(args.seed)

    # Data processing
    logger.info("Process data")
    train_data_path = args.data_path
    with open(train_data_path, "rb") as f:
        train_b1_2 = np.load(f, allow_pickle=True)
    train_b1_2 = train_b1_2.reshape(-1, 256).astype(float)
    time_2 = np.linspace(0.001, 145.870000, train_b1_2.shape[0] * 16)
    time_2 = time_2.reshape(-1, 16)
    train_b1_2 = np.concatenate((train_b1_2, time_2), axis=1)
    # Column information
    NUMERIC_FEATURES = [f"f{i}" for i in range(256)]
    TIME_FEATURES = [f"t{i}" for i in range(16)]
    FEATURES = list(NUMERIC_FEATURES) + list(TIME_FEATURES)
    LABEL = "rul"
    train_data = pd.DataFrame(data=train_b1_2, columns=FEATURES)
    train_data[LABEL] = np.array(list(np.linspace(1.00, 0.001, train_b1_2.shape[0])))
    train_data[LABEL].mean()
    train_data[FEATURES] = train_data[FEATURES].astype(float)
    # Train/test split
    X_train, X_val = train_test_split(train_data, test_size=0.2)
    sc = StandardScaler()
    X_train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(X_train[NUMERIC_FEATURES])
    X_val.loc[:, NUMERIC_FEATURES] = sc.transform(X_val[NUMERIC_FEATURES])
    train_dataset = df_to_dataset(
        X_train[FEATURES + [LABEL]], LABEL, batch_size=args.batch_size
    )
    val_dataset = df_to_dataset(
        X_val[FEATURES + [LABEL]], LABEL, batch_size=args.batch_size, shuffle=False
    )  # No shuffle

    optimizer = tfa.optimizers.AdamW(
        learning_rate=args.lr, weight_decay=args.weight_decay
    )
    # Create model
    logger.info("Create model")
    model = create_model(
        time_features=TIME_FEATURES,
        numerical_features=NUMERIC_FEATURES,
        numerical_data=X_train[NUMERIC_FEATURES].values,
        optimizer=optimizer,
    )

    # Train
    early = EarlyStopping(
        monitor="val_output_loss", mode="min", patience=20, restore_best_weights=True
    )
    callback_list = [early]
    ft_linear_history = model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        callbacks=callback_list,
    )

    # Model summary
    model.summary()

    # Evaluate
    linear_val_preds = model.predict(val_dataset)
    RMSE = np.sqrt(
        mean_squared_error(X_val[LABEL].values, linear_val_preds["output"].ravel())
    )
    logger.info(f"RMSE {RMSE}")


if __name__ == "__main__":
    main()
