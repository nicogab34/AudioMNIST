import glob

def get_epoch_checkpoint(checkpoint_path, epoch):
    epoch_checkpoint = glob.glob(os.path.join(checkpoint_path, f"model.{epoch}-*.data*"))
    assert len(epoch_checkpoint) == 1
    epoch_checkpoint = epoch_checkpoint[0].split(".data")[0]
    return epoch_checkpoint