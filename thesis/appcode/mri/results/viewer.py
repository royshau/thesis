# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import json
import copy
import sys
import matplotlib.patches as patches
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'
BRAIN_SUFFIX = '_brain'
# IMG_FORMAT = 'eps'
IMG_FORMAT = 'png'
ROI_FORMAT = 'roi'
MASK_FORMAT = 'mask'
DPI = 250
CLASSES = [0,1,2,3]


class Viewer(object):
    def __init__(self, data_dir, output_dir='/tmp/', num_of_cases=-1, suffixes=None, brain_only=True, views=None,
                 mask=None, roi=None, first=None):
        """
        Viewer for post training, compare examples and dump images for paper
        :param data_dir:
        :param num_of_cases:
        :param suffixes:
        :param brain_only:
        :param views:
        """
        self.data_dir = data_dir
        self.num_of_cases = 1000 if num_of_cases == -1 else num_of_cases
        self.suffixes = eval(suffixes) if suffixes is not None else None
        self.brain_only = brain_only
        self.views = eval(views) if views is not None else ['images', 'brains', 'segs']
        self.seg_suffix_use = BRAIN_SUFFIX+SEG_SUFFIX if brain_only else SEG_SUFFIX
        self.sub_dirs = os.listdir(data_dir)
        self.output_dir = output_dir
        self.counter = 0
        self.data = None
        self.data_mask = None
        self.mask = eval(mask) if mask is not None else None
        self.roi = eval(roi) if roi is not None else None
        self.first = 0 if first is None else first

        # if self.mask is not None and self.roi is not None:
        #     self.mask['x1'] = self.roi['x1']

    def show(self, case_name=None):
        """
        Show function
        :param case_name:
        :return:
        """
        self.counter = 1
        for sub_dir in self.sub_dirs:
            if self.sub_dirs.index(sub_dir) > self.num_of_cases - 1:
                break
            if not os.path.isdir(os.path.join(self.data_dir, sub_dir)):
                continue
            # sub_dir = '022-Guys-0701-T1'
            # Load one case and store it in self.data
            self.load_one_case(sub_dir)

            # View one case
            ret = self.view_case(case_name=sub_dir)

            # Exit condition
            if ret == 0:
                return

            self.counter += 1

    def load_one_case(self, sub_dir):
        """
        Load one case
        :param sub_dir: 
        :return: 
        """
        path_sub_dir = os.path.join(self.data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)

        suffix_to_nifti_brain = {}
        suffix_to_nifti_seg = {}
        suffix_to_nifti_image = {}

        print("Working on - %s, number: (%d / %d)" % (sub_dir, self.counter, self.num_of_cases))
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in self.suffixes:
                if file_type.endswith(sub_dir+suffix+self.seg_suffix_use+NII_SUFFIX):
                    suffix_to_nifti_seg[suffix] = nib.load(os.path.join(path_sub_dir, file_type)).get_data()
                    brain_path = os.path.join(path_sub_dir, file_type.replace('_seg', ''))
                    suffix_to_nifti_brain[suffix] = nib.load(brain_path).get_data()
                    suffix_to_nifti_image[suffix] = nib.load(brain_path.replace('_brain', '')).get_data()

        self.data = {}
        self.data_mask = {}
        if 'images' in self.views:
            self.data['images'] = copy.deepcopy(suffix_to_nifti_image)
            self.data_mask['images'] = copy.deepcopy(suffix_to_nifti_image)
        if 'brains' in self.views:
            self.data['brains'] = copy.deepcopy(suffix_to_nifti_brain)
            self.data_mask['brains'] = copy.deepcopy(suffix_to_nifti_brain)
        if 'segs' in self.views:
            self.data['segs'] = copy.deepcopy(suffix_to_nifti_seg)
            self.data_mask['segs'] = copy.deepcopy(suffix_to_nifti_seg)

        # Slice according to given mask
        if self.mask is not None or self.roi is not None:
            for view in self.data.keys():
                for suffix in self.data[view].keys():
                    if self.roi is not None:
                        self.data[view][suffix] = \
                            self.data[view][suffix][self.roi['y1']: self.roi['y2'], self.roi['x1']: self.roi['x2'], :]
                    if self.mask is not None:
                        self.data_mask[view][suffix] = \
                            self.data[view][suffix][self.mask['y1']: self.mask['y2'], self.mask['x1']: self.mask['x2'], :]

    def dump_example(self, case_name, view, suffix, i, fig, extent):
        if not os.path.exists(os.path.join(self.output_dir, case_name)):
            os.makedirs(os.path.join(self.output_dir, case_name))
        out_file = os.path.join(self.output_dir, case_name, '%s_%s_%s_%d.%s' %
                                (case_name, view, suffix, i[0], IMG_FORMAT))
        fig.savefig(out_file, bbox_inches=extent, format=IMG_FORMAT, dpi=DPI)
        # Dump some information regarding ROI and MASK
        if self.roi is not None:
            out_file = os.path.join(self.output_dir, case_name, '%s_%s_%s_%d.%s' %
                                    (case_name, view, suffix, i[0], ROI_FORMAT))
            with open(out_file, 'w') as f:
                json.dump(self.roi, f)
        if self.mask is not None:
            out_file = os.path.join(self.output_dir, case_name, '%s_%s_%s_%d.%s' %
                                    (case_name, view, suffix, i[0], MASK_FORMAT))
            with open(out_file, 'w') as f:
                json.dump(self.mask, f)

    def view_case(self, case_name):
        """
        Show one case stored in self.data
        """
        suffixes = self.data[self.data.keys()[0]].keys()
        views = self.data.keys()

        give_me_more = [True]
        val_to_ret = [1]
        save_example = [False]

        i = [self.first]

        def on_key(event):
            key = event.key
            if key == 'control' or key == 'alt' or key == 'ctrl+alt':
                return
            print 'you pressed %r' % key
            if key == ' ':
                i[0] += 1
                # call the draw_samples again with next images
            elif key == 'd':
                i[0] = i[0]
                save_example[0] = True
            elif key == 'g':
                age = input("Please choose slice - ")
                i[0] = int(age)
                print("Move to slice %d " % i[0])
            elif key == 'p':
                i[0] = max(i[0] - 1, 1)
            if key == ']':
                i[0] += 10
                # call the draw_samples again with next images
            elif key == '[':
                i[0] = max(i[0] - 10, 1)
            elif key == 'q' or key == 'x':
                plt.close(fig)
                give_me_more[0] = False
                val_to_ret[0] = 1 if key == 'q' else 0
                return

        if self.mask is not None:
            fig, ax = plt.subplots(nrows=len(views)+1, ncols=len(suffixes))
        else:
            fig, ax = plt.subplots(nrows=len(views), ncols=len(suffixes))

        if len(views) == 1 and len(suffixes) == 1:
            # ax = np.array(ax).reshape(1,1)
            ax = ax.reshape(2, 1) if self.mask is not None else ax.reshape(1, 1)
        else:
            if len(views) == 1:
                ax = ax.reshape(2, -1) if self.mask is not None else ax.reshape(1, -1)
            if len(suffixes) == 1:
                ax = ax.reshape(-1, 2) if self.mask is not None else ax.reshape(-1, 1)

        # fig.tight_layout()
        fig.set_size_inches(18.5, 10.5, forward=True)

        while give_me_more[0]:
            for view in views:
                ax_row = views.index(view)
                for suffix in suffixes:
                    ax_col = suffixes.index(suffix)

                    ax[ax_row][ax_col].set_title("%s, i = %d" % (suffix, i[0]))
                    cmap = 'gray' if view != 'segs' else plt.get_cmap('YlGnBu')

                    ax[ax_row][ax_col].imshow(self.data[view][suffix][:, :, i[0]],
                                              interpolation="none", cmap=cmap)
                    if self.mask is not None:
                        ax[ax_row][ax_col].add_patch(
                            patches.Rectangle(
                            (self.mask['x1'], self.mask['y1']),   # (x,y)
                            self.mask['x2']-self.mask['x1'],      # width
                            self.mask['y2'] - self.mask['y1'],    # height
                            fill=False,
                            color='red',
                            linewidth=3.0
                            )
                        )
                        ax[ax_row+1][ax_col].imshow(self.data_mask[view][suffix][:, :, i[0]],
                                                  interpolation="none", cmap=cmap)
                        ax[ax_row+1][ax_col].axis('off')

                    ax[ax_row][ax_col].axis('off')

            print("Viewing case: %s, slice: %d" % (case_name, i[0]))
            plt.draw()

            fig.canvas.mpl_connect('key_press_event', on_key)
            sys.stdout.flush()
            if not plt.waitforbuttonpress():
                # fig.tight_layout()
                pass

            if save_example[0]:
                save_example[0] = False
                print("Save example")

                # fig.savefig(out_fig, format='eps', dpi=1000)
                for view in views:
                    ax_row = views.index(view)
                    for suffix in suffixes:
                        ax_col = suffixes.index(suffix)
                        extent = ax[ax_row][ax_col].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        self.dump_example(case_name, view, suffix, i, fig, extent)
                        if self.mask is not None:
                            ax_col = suffixes.index(suffix)
                            extent = ax[ax_row+1][ax_col].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                            self.dump_example(case_name, view, suffix+"_zoom", i, fig, extent)

        print('Move to new case')
        return val_to_ret[0]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='/tmp/', help='output directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    parser.add_argument('--views', dest='views', default=None, type=str, help='views')
    parser.add_argument('--brain_only', dest='brain_only', default=True, type=bool, help='brain only')
    parser.add_argument('--roi', dest='roi', default=None, type=str, help='roi')
    parser.add_argument('--mask', dest='mask', default=None, type=str, help='mask')
    parser.add_argument('--first', dest='first', default=None, type=int, help='first slice')
    args = parser.parse_args()

    viewer = Viewer(data_dir=args.data_dir, output_dir=args.output_dir, num_of_cases=args.num_of_cases,
                    suffixes=args.suffixes, brain_only=args.brain_only, views=args.views, mask=args.mask, roi=args.roi, first=args.first)
    viewer.show()
