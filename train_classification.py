import argparse

from audiomnist.io.read_dataset import load_raw_dataset, load_spectrogram_dataset
from audiomnist.train.dataset_preparation import prepare_datasets_digit_cls
from audiomnist.train.train import train
from tensorflow.keras import optimizers
from importlib import import_module

DATASET_LOAD = {
    "audionet" : load_raw_dataset,
    "audionet_big" : load_raw_dataset,
    "alexnet" : load_spectrogram_dataset
}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Training script for tensorflow.keras AudioNet model.")
    parser.add_argument('model', help="Model to train")
    parser.add_argument('-i','--input_dataset', help="path to TFRecord file", required=True)
    parser.add_argument('-o','--checkpoint_output', help="path to checkpoint folder", required=True)
    parser.add_argument('-l','--logdir', help="path to logdir", required=True)
    parser.add_argument('-b','--batch_size', help="Batch size", required=True, type=int)
    parser.add_argument('-e','--epochs', help="Epochs", required=True, type=int)
    parser.add_argument('-lr','--learning_rate', help="Learning rate", required=True, type=float)

    args = parser.parse_args()

    model_pkg = import_module(f"audiomnist.models.{args.model}")

    dataset = DATASET_LOAD[args.model](args.input_dataset)
    datasets, sizes = prepare_datasets_digit_cls(dataset, args.batch_size)

    model = model_pkg.build_model()
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=optimizers.Adam(lr=args.learning_rate))

    train(model, datasets, sizes, args.checkpoint_output, args.logdir, args.batch_size, args.epochs)