import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import numpy as np
from torch.serialization import save

from utils import NUMBER_PERSONS


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def draw_solo(saved_pred, wto):

    # all on cpu
    num_env = len(saved_pred)
    for i in range(num_env):
        saved_pred[i] = [x.cpu() for x in saved_pred[i]]

    X = 1
    Y = len(saved_pred)//X+1 if len(saved_pred)%X != 0 else  len(saved_pred)//X
    # create the plot 
    figure, axes = plt.subplots(X, Y)
    figure.set_size_inches(2*Y, 2*X)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    num_seq = saved_pred[0][0].shape[1]//NUMBER_PERSONS
    a = np.arange(num_seq)

    for k, (obs, fut, pred) in enumerate(saved_pred):
      for i, seq in enumerate(a[wto:wto+1]):
          for j in range(NUMBER_PERSONS):
              axes[k//X].plot(obs[:,NUMBER_PERSONS*seq+j,0], obs[:,NUMBER_PERSONS*seq+j,1], label='obs', color=colors[j])
              axes[k//X].plot(fut[:,NUMBER_PERSONS*seq+j,0], fut[:,NUMBER_PERSONS*seq+j,1], label='fut', color=colors[j])
              axes[k//X].plot(pred[:,NUMBER_PERSONS*seq+j,0], pred[:,NUMBER_PERSONS*seq+j,1], '--', label='pred', color=colors[j])

            
                  # axes[k%X][k//X].plot(obs[:,NUMBER_PERSONS*seq+j,0], obs[:,NUMBER_PERSONS*seq+j,1], label='obs', color=colors[j])
                  # axes[k%X][k//X].plot(fut[:,NUMBER_PERSONS*seq+j,0], fut[:,NUMBER_PERSONS*seq+j,1], label='fut', color=colors[j])
                  # axes[k%X][k//X].plot(pred[:,NUMBER_PERSONS*seq+j,0], pred[:,NUMBER_PERSONS*seq+j,1], '--', label='pred', color=colors[j])

    # convert it to numpy array
    cm_image = plot_to_image(figure)
    array = (cm_image.numpy()[0])[:,:,:3]
    array = np.transpose(array, (2, 0, 1))

    return figure, array

def draw_solo_all(saved_pred):

    # all on cpu
    for i in range(len(saved_pred)):
      for j in range(len(saved_pred[0])):
        saved_pred[i][j] = [x.cpu() for x in saved_pred[i][j]]
    
    X = len(saved_pred)
    Y = len(saved_pred[0])
    # create the plot 
    figure, axes = plt.subplots(X, Y)
    figure.set_size_inches(2*Y, 2*X)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    num_seq = saved_pred[0][0][0].shape[1]//NUMBER_PERSONS
    a = np.arange(num_seq)

    for m, saved_pred_ in enumerate(saved_pred):
      for k, (obs, fut, pred) in enumerate(saved_pred_):
        for i, seq in enumerate(a[m:m+1]):
            for j in range(NUMBER_PERSONS):
                axes[m][k].set_xticks([])
                axes[m][k].set_yticks([])
                axes[m][k].plot(obs[:,NUMBER_PERSONS*seq+j,0], obs[:,NUMBER_PERSONS*seq+j,1], label='obs', color=colors[j])
                axes[m][k].plot(fut[:,NUMBER_PERSONS*seq+j,0], fut[:,NUMBER_PERSONS*seq+j,1], label='fut', color=colors[j])
                axes[m][k].plot(pred[:,NUMBER_PERSONS*seq+j,0], pred[:,NUMBER_PERSONS*seq+j,1], '--', label='pred', color=colors[j])

    # convert it to numpy array
    cm_image = plot_to_image(figure)
    array = (cm_image.numpy()[0])[:,:,:3]
    array = np.transpose(array, (2, 0, 1))

    return figure, array




def draw_image(saved_pred):

    # all on cpu
    num_env = len(saved_pred)
    for i in range(num_env):
        saved_pred[i] = [x.cpu() for x in saved_pred[i]]

    # create the plot 
    figure, axes = plt.subplots(3, 2*num_env)
    figure.set_size_inches(4*num_env, 6)
    colors = ['red', 'blue', 'green', 'black', 'yellow', 'purple']

    num_seq = saved_pred[0][0].shape[1]//NUMBER_PERSONS
    a = np.arange(num_seq)
    # np.random.shuffle(a)

    for k, (obs, fut, pred) in enumerate(saved_pred):
      for i, seq in enumerate(a[:6]):
          for j in range(NUMBER_PERSONS):
                  axes[i%3][i//3+2*k].set_xticks([])
                  axes[i%3][i//3+2*k].set_yticks([])
                  axes[i%3][i//3+2*k].plot(obs[:,NUMBER_PERSONS*seq+j,0], obs[:,NUMBER_PERSONS*seq+j,1], label='obs', color=colors[j])
                  axes[i%3][i//3+2*k].plot(fut[:,NUMBER_PERSONS*seq+j,0], fut[:,NUMBER_PERSONS*seq+j,1], label='fut', color=colors[j])
                  # axes[i%3][i//3+2*k].plot(pred[:,NUMBER_PERSONS*seq+j,0], pred[:,NUMBER_PERSONS*seq+j,1], '--', label='pred', color=colors[j])



    # for k, (obs, fut, pred) in enumerate(saved_pred):
    #     for i, seq in enumerate(a[:6]):
    #         for j in range(4):
    #                 axes[i%3][i//3+2*k].plot(obs[:,seq+j,0], obs[:,seq+j,1], label='obs', color=colors[j])
    #                 axes[i%3][i//3+2*k].plot(fut[:,seq+j,0], fut[:,seq+j,1], label='fut', color=colors[j])
    #                 axes[i%3][i//3+2*k].plot(pred[:,seq+j,0], pred[:,seq+j,1], '--', label='pred', color=colors[j])

    # convert it to numpy array
    cm_image = plot_to_image(figure)
    array = (cm_image.numpy()[0])[:,:,:3]
    array = np.transpose(array, (2, 0, 1))

    return figure, array
