import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def imshow(image, colormap='gray', ax=None, fig=None):
    """
    Image show
    :param image: image to show
    :param colormap: colormap
    :param ax: axes handle
    :param fig: figure
    :return: plot
    """
    if ax and fig:
        # For subplots
        img = ax.imshow(image, interpolation="none", cmap=plt.get_cmap(colormap))
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="10%", pad=0.05)
        plt.colorbar(img, cax=cax)
        plt.show(block=False)
    else:
        # For only one plot, good for interactive mode
        plt.imshow(image, interpolation="none", cmap=plt.get_cmap(colormap))
        plt.colorbar()
        plt.show(block=False)
