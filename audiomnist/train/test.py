import pandas as pd
from .checkpoint import get_epoch_checkpoint

def test_best_epoch(model, datasets, sizes, checkpoint_path, batch_size):
    history = pd.from_csv(os.path.join(checkpoint_path, "history.csv"))

    best_epoch = history.val_acc.idxmax() + 1

    best_epoch_checkpoint = get_epoch_checkpoint(checkpoint_path, best_epoch)

    test(model, datasets, sizes, checkpoint_path, best_epoch_checkpoint, batch_size)

def test(model, datasets, sizes, checkpoint_path, epoch, batch_size):

    epoch_checkpoint = get_epoch_checkpoint(checkpoint_path, epoch)

    model.load_weights(epoch_checkpoint)

    scores = model.evaluate(datasets["test"], steps=int(math.ceil(sizes["test"]/batch_size)))

    with open(os.path.join(checkpoint_path,f"evalution_epoch{epoch}.txt"), "w") as fh:
        for i, name in enumerate(model.metrics_names):
            fh.write(f"{name} : {scores[i]}\n")