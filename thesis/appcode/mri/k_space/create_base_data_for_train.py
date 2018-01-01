import numpy as np
import argparse


from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.data_creator import DataCreator


def create_base_data_for_train(data_base_name, output_dir, axial_limits, debug, trans=None, rot=0):
    """
    Create data for training form dataBase name.
    The scripts create directory for each case in train/test and write each example
    in separate file with a counter.
    :param data_base_name: For example, 'SchizRef'
    :param output_dir: Base output dir
    :param axial_limits: limits for axial axis, default : [15, 90]
    :param debug: debug mode
    :param trans: transpose
    :param rot: rotation angle: 0,90,180,270
    :return:
    """

    axial_limits = np.array(eval(axial_limits))
    trans = eval(trans) if len(trans) > 0 else None

    data_source = MriDataBase(data_base_name)
    data_creator = DataCreator(data_source, output_dir, axial_limits=axial_limits, rot=rot)
    data_creator.create_examples(debug=debug, trans=trans)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_base_name', required=True, type=str, help='Name of database, for example ''SchizRef''')
    # example : 'SchizReg'

    parser.add_argument('--output_dir', required=True, type=str, help='Basic output directory for data base')
    # example :output_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/03_01_2016/base/'

    parser.add_argument('--axial_limits', required=False, default='[15, 100]', type=str, help='axial limits')
    # example: np.array([15, 100])

    parser.add_argument('--debug', required=False, default=False, type=bool, help='debug mode')

    parser.add_argument('--trans', required=False, default='', type=str, help='Transpose')
    parser.add_argument('--rot', required=False, default=0, type=int, help='rotation')

    args = parser.parse_args()

    # Run script:
    create_base_data_for_train(args.data_base_name, args.output_dir, args.axial_limits, debug=args.debug,
                               trans=args.trans, rot=args.rot)

    # # Debug
    # create_base_data_for_train('SchizReg', '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/03_01_2016/base/'
    #                            , '[15, 100]')