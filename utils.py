"""
Utility functions and methods
"""
import matplotlib.pyplot as plt

def plot_loss(train_loss_list,validation_loss_list):
    """
    Plot training and validation loss curves.
    Parameters
    ----------
    `train_loss_list` : Sequence[torch.Tensor | numbers.Number]
        Iterable of training loss values collected during training. Elements that are
        torch tensors will be converted with `.cpu().detach()` before plotting.    
    `validation_loss_list` : Sequence[torch.Tensor | numbers.Number]
        Iterable of validation loss values collected during training. Elements that are
        torch tensors will be converted with `.cpu().detach()` before plotting.
    Returns
    -------
    None
        Displays a matplotlib figure showing the loss curves. No value is returned.
    Behavior
    --------
    - Plots training loss in green and validation loss in red.
    - X-axis corresponds to the index of each loss entry (e.g., logging steps such as "every 100 epochs").
    - Adds axis labels and a legend, then calls `plt.show()` to display the plot.
    Notes
    -----
    - This function expects `matplotlib.pyplot` to be available as `plt`.
    - If elements are neither tensors nor numeric types, attempting to call `.cpu()` or `.detach()`
      will raise an AttributeError.
    - For best visual comparison, provide sequences of comparable length or preprocess accordingly.
    """
    train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
    validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

    plt.plot(train_loss_list_converted,'g',label='train_loss')
    plt.plot(validation_loss_list_converted,'r',label='validation_loss')
    plt.xlabel('Steps - every 100 epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()