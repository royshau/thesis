import numpy as np


class FileHandler:
    """
    Describes file object with specific info type.
    This class can create a file and write into it,
    also, can read this kind of file, reshape it , etc.
    """

    def __init__(self, path, info, read_or_write, name=None, memmap=False):
        self.path = path
        self.info = info
        self.name = name
        self.file_mode = read_or_write
        self.block_size = info["width"] * info["height"] * self.info["channels"]
        if memmap:
            self.memmap = self.reshape(np.memmap(self.path, dtype=self.info['dtype'], mode='r'))
        else:
            self.file_obj = self.__create_file_obj__()


    def get_file_obj(self):
        """
        Accessor to file object
        :return: file object
        """
        return self.file_obj

    def __create_file_obj__(self):
        """
        Create file object for writing or reading
        :return: file object
        """
        file_mode = 'rb' if self.file_mode == 'read' else 'wb'
        self.file_obj = open(self.path, file_mode)
        return self.file_obj

    def write(self, data):
        """
        Write data
        :param data: numpy array. need to be: (Examples, height, width)
        :return: none
        """
        assert self.file_mode == 'write'
        assert self.file_obj is not None
        assert self.info["dtype"] == data.dtype
        self.file_obj.write(data.ravel())

    def read(self, n=-1, reshaped=True):
        """
        Read n examples
        :param n: number of examples
        :param reshaped: reshape blob
        :return:
        """
        assert self.file_mode == 'read'
        assert self.file_obj is not None
        ret = np.fromfile(self.file_obj, count=n*self.block_size, dtype=self.info["dtype"])

        if reshaped:
            ret = self.reshape(ret)

        return ret

    def reshape(self, data):
        """
        Reshape blob
        :param data: numpy array
        :return: reshaped data
        """
        n = len(data)
        ch = self.info["channels"]
        if ch == 1:
            ret = data.reshape(n / self.block_size, self.info['width'], self.info['height']).transpose().squeeze()
            # N, h, w
            ret = ret.transpose(np.roll(np.arange(len(ret.shape)), 1))
        else:
            ret = data.reshape(n / self.block_size, self.info['width'], self.info['height'], ch).transpose().squeeze()
            # N, h, w
            ret = ret.transpose(np.roll(np.arange(len(ret.shape)), 1))

        return ret

