import argparse
from audiomnist.train import autoencoder_spectrogram

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Training script for autoencoder model.")
    parser.add_argument('-i','--input_dataset', help="path to TFRecord file", required=True)
    parser.add_argument('-o','--checkpoint_output', help="path to checkpoint folder", required=True)
    parser.add_argument('-l','--logdir', help="path to logdir", required=True)
    parser.add_argument('-b','--batch_size', help="Batch size", required=True, type=int)
    parser.add_argument('-e','--epochs', help="Epochs", required=True, type=int)

    args = parser.parse_args()

    autoencoder_spectrogram.train(args.input_dataset, args.checkpoint_output, args.logdir, args.batch_size, args.epochs)