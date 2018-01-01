import os
import numpy as np
from appcode.mri.k_space.files_info import get_info
from common.files_IO.file_handler import FileHandler


class KspaceDataSet:
    """
    Kspace data base class
    """

    def __init__(self, base_dir, file_names, stack_size=500, shuffle=True, data_base='SchizReg', memmap=False):
        """
        Constructor
        :param base_dir: basic dir after shuffle
        :param file_names: handle only specific file names
        :param stack_size: stack size of examples
        :param shuffle: read data with shuffle or not
        :return:
        """
        self.path = base_dir
        self.file_names = file_names
        self.files_info = get_info(data_base)
        self.train = DataSet(os.path.join(base_dir, "train"), file_names, stack_size,
                             shuffle=shuffle, files_info=self.files_info, memmap=memmap)
        self.test = DataSet(os.path.join(base_dir, "test"), file_names, stack_size,
                            shuffle=shuffle, files_info=self.files_info, memmap=memmap)
        self.stack_size = stack_size


class DataSet:
    """
    Train / Test Data Set
    """

    def __init__(self, base_dir, file_names, stack_size, shuffle=True, files_info=None, memmap=False):
        """
        Constructor
        :param base_dir: basic dir after shuffle
        :param file_names: handle only specific file names
        :param stack_size: stack size of examples
        :param shuffle: read data with shuffle or not
        :return:
        """
        self.path = base_dir
        self.file_names = file_names
        self.counter = -1     # Counter of examples, used as offset
        self.epoch = 0  # count the number of times which we read all the data
        self.current = {name: None for name in file_names}  # Current data holder
        self.N_MAX = stack_size  # Number of examples in current
        self.files_obj = {}
        self.shuffle = shuffle
        self.files_info = files_info
        # Init all files objects, for reading
        self.init_files_obj(memmap)

    def init_files_obj(self, memmap=False):
        """
        Init file objects, update pointers to start of the file
        :return:
        """
        files_obj = {}
        for (file_name, info) in self.files_info.iteritems():
            if file_name in self.file_names:
                path = os.path.join(self.path, "000000.%s.bin" % file_name)
                files_obj[file_name] = FileHandler(path, info, "read", name=None, memmap=memmap)

        self.files_obj = files_obj

    def update_current(self):
        """
        Update data base with N new examples
        :return:
        """
        for file_name in self.file_names:
            data = self.files_obj[file_name].read(n=self.N_MAX, reshaped=True)

            # No more data, EOF
            if data.shape[0] == 0:
                # Re-init the file objects, and call update current again
                self.init_files_obj()
                self.epoch += 1
                return self.update_current()

            self.current[file_name] = data

        self.counter = 0

        # Get permutation
        perm = np.arange(data.shape[0])

        if self.shuffle:
            np.random.shuffle(perm)

        # Re-ordering
        for file_name in self.file_names:

            # Reorder according to dimension
            if len(self.current[file_name].shape) == 2:
                self.current[file_name] = self.current[file_name][perm, :]
            elif len(self.current[file_name].shape) == 3:
                self.current[file_name] = self.current[file_name][perm, :, :]
            elif len(self.current[file_name].shape) == 4:
                self.current[file_name] = self.current[file_name][perm, :, :, :]

    def next_batch(self, n, norm=False):
        """
        Return n examples
        :param n: Number of examples
        :return: numpy.array
        """
        # First, n must be small then self.N
        assert n <= self.N_MAX

        # If not enough examples, update current state
        # TODO: add only missing examples
        if (n + self.counter > self.N_MAX) or self.counter == -1:
            self.update_current()
            return self.next_batch(n)

        ret = {}
        for file_name in self.current.keys():

            offset = self.counter
            if len(self.current[file_name].shape) == 2:
                ret[file_name] = self.current[file_name][offset:offset+n, :]
            elif len(self.current[file_name].shape) == 3:
                ret[file_name] = self.current[file_name][offset:offset+n, :, :]
            elif len(self.current[file_name].shape) == 4:
                ret[file_name] = self.current[file_name][offset:offset+n, :, :, :]

        # Update counter
        self.counter += n
        return ret
