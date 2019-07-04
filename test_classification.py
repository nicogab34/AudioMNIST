import argparse
from audiomnist.io.read_dataset import load_raw_dataset, load_spectrogram_dataset
from audiomnist.train.dataset_preparation import prepare_datasets_digit_cls
from importlib import import_module

DATASET_LOAD = {
    "audionet" : load_raw_dataset,
    "audionet_big" : load_raw_dataset,
    "alexnet" : load_spectrogram_dataset
}

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Testing script for tensorflow.keras AudioNet model.")
    parser.add_argument('model', help="Model to train")
    parser.add_argument('-i','--input_dataset', help="path to TFRecord file", required=True)
    parser.add_argument('-o','--checkpoint_output', help="path to checkpoint folder", required=True)
    parser.add_argument('-e','--epoch', help="epoch to test, if none, the best is evaluated")
    parser.add_argument('-b','--batch_size', help="Batch size", required=True, type=int)


    args = parser.parse_args()

    dataset = DATASET_LOAD[args.model](args.input_dataset)
    datasets, sizes = prepare_datasets_digit_cls(dataset)

    model_pkg = import_module(f"audiomnist.models.{args.model}")

    model = model_pkg.build_model()
    model.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer="adam")

    if epoch:
        test(model, datasets, sizes, checkpoint_path, epoch, batch_size)
    else:
        test_best_epoch(model, datasets, sizes, checkpoint_path, batch_size)