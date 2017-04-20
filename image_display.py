import matplotlib.pyplot as plt

def side_by_side_plot(im1, im2, im1_title=None, im2_title=None, im1_cmap=None, im2_cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    f.tight_layout()
    ax1.imshow(im1, cmap=im1_cmap)
    if im1_title: ax1.set_title(im1_title, fontsize=30)
    ax2.imshow(im2, cmap=im2_cmap)
    if im2_title: ax2.set_title(im2_title, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()