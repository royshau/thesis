import numpy as np
import matplotlib.pyplot as plt
import sys

# sys.path.append("/home/ohadsh/work/python/thesis/")

from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.utils import get_dummy_k_space_and_image, get_image_from_kspace, pad_image_with_zeros_fixed
from common.viewers.imshow import imshow


to_show = True

# data_source = MriDataBase('SchizReg')
# data_source = MriDataBase('ADNI_Normal')
# data_source = MriDataBase('IXI_T1')
# data_source = MriDataBase('IXI_T2')
data_source = MriDataBase('OASIS_AD')

# k-space creation and view

# item = '116_T_1232/ID_116_T_1232_Original.nii.gz'
data_example = data_source.get_source_data(data_source.items[0])
# data_example = data_source.get_source_data(item)
# dat = data_example['img'][0].squeeze().transpose(0,2,1)
dat = data_example['img'][0].squeeze().transpose(1,2,0)
# dat = data_example['img'][0].squeeze()

# dat = np.rot90(np.rot90(dat))

print dat.shape

# dat = pad_image_with_zeros_fixed(dat=dat, to_size=[256, 256])
print ("min/max = (%f, %f)" % (dat.min(), dat.max()))


# k_space, dummy_image = get_dummy_k_space_and_image(dat[:,:,50])
k_space_all, dummy_image_all = get_dummy_k_space_and_image(dat)


if to_show:

    fig, ax = plt.subplots(nrows=2, ncols=2)

    for i in range(30, 256, 5):
        original_image = dat[:, :, i]
        k_space = k_space_all[:, :, i]
        dummy_image = dummy_image_all[:, :, i]
        rec_image = get_image_from_kspace(k_real=k_space.real, k_imag=k_space.imag)

        ax[0][0].set_title('Original Image')
        imshow(original_image, ax=ax[0][0], fig=fig, draw_colorbar=False)

        ax[0][1].set_title('Log-k_space')
        imshow(np.log(np.abs(k_space)), ax=ax[0][1], fig=fig, draw_colorbar=False)

        ax[1][0].set_title('Dummy Image')
        imshow(rec_image, ax=ax[1][0], fig=fig, draw_colorbar=False)

        ax[1][1].set_title('Diff')
        imshow(np.abs(original_image - rec_image), ax=ax[1][1], fig=fig, draw_colorbar=False)

        plt.tight_layout()
        plt.waitforbuttonpress()
a = 1