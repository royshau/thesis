# import tensorflow as tf
# predict_d_logits = 0
# predict_d_logits_for_g = 0
#
# d_loss_real = tf.reduce_mean(predict_d_logits)
# d_loss_fake = tf.reduce_mean(predict_d_logits_for_g)
# d_loss = d_loss_fake - d_loss_real
# g_loss = -tf.reduce_mean(predict_d_logits_for_g)

import matplotlib.pyplot as plt
from appcode.mri.k_space.data_creator import get_random_mask, get_random_gaussian_mask, get_rv_mask
mask_single = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=2)
mask_single4 = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=4)
mask_single6 = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=6)

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(mask_single,interpolation="none", cmap="gray")
ax[0].axis('off')
# ax[0].set_title('Factor 2.5 - 40%')
ax[1].imshow(mask_single4,interpolation="none", cmap="gray")
ax[1].axis('off')
# ax[1].set_title('Factor 4 - 25%')
ax[2].imshow(mask_single6,interpolation="none", cmap="gray")
ax[2].axis('off')
# ax[2].set_title('Factor 6 - 16%')


extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('/tmp/mask2.5.eps', bbox_inches=extent, format='eps', dpi=1000)
extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('/tmp/mask4.eps', bbox_inches=extent, format='eps', dpi=1000)
extent = ax[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('/tmp/mask6.eps', bbox_inches=extent, format='eps', dpi=1000)

plt.show()