import math
import os
import tensorflow as tf
import gc
import numpy as np

def train(model, datasets, sizes, checkpoint_path, logdir, batch_size, epochs):

    if not os.path.isdir(logdir): os.mkdir(logdir)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                 batch_size=batch_size)

    if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, "model.{epoch:02d}-{val_acc:.2f}"),
                                                            save_weights_only=True)

    gc_callback = tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch,_: gc.collect())

    history = model.fit(datasets["train"],
              epochs=epochs,
              steps_per_epoch=int(math.ceil(sizes["train"]/batch_size)),
              validation_data=datasets["validation"],
              validation_steps=int(math.ceil(sizes["validation"]/batch_size)),
              shuffle=False,
              callbacks=[tb_callback, checkpoint_callback])

    hist_df = pd.DataFrame.from_dict(history.history)

    hist_df.to_csv(os.path.join(checkpoint_path, "history.csv"))