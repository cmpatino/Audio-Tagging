import sys
import pickle
import matplotlib.pyplot as plt


def create_plot(history_file, title):
    """Creates plots of learning curves for train and validation sets
    
    Arguments:
        history_file {str} -- filepath to binary file with loss history
        title {str} -- string to use as plot title
    """

    with open(history_file, 'rb') as f:
        training_dict = pickle.load(f)

    n_epochs = len(training_dict['loss'])

    with plt.style.context('seaborn-talk'):

        ax = plt.axes()
        ax.plot(training_dict['loss'], label='Training Loss', marker='^')
        ax.plot(training_dict['val_loss'], label='Val Loss', marker='s')
        ax.xaxis.set_major_locator(plt.MaxNLocator(n_epochs))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, y: int(x + 1)))
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=15)
        plt.title(title, fontsize=25, y=1.05)
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend()
        plt.savefig('../visualizations/' + title + '_loss.png')


if __name__ == '__main__':

    history_file = sys.argv[1]
    title = sys.argv[2]

    create_plot(history_file, title)
