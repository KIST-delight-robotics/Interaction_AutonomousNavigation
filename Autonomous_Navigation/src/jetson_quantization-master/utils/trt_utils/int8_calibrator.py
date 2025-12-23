import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import os
import random

class DataLoader:
    def __init__(self, batch_size, calib_count, calib_files_dir, shuffle=False):
        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.file_list = [os.path.join(calib_files_dir, f) for f in os.listdir(calib_files_dir) if f.endswith('.npy')]
        assert len(self.file_list) >= self.batch_size * self.calib_count
        self.shuffle = shuffle

        if self.shuffle:
            random.shuffle(self.file_list)  # 파일 리스트를 셔플

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            data = np.load(self.file_list[self.index], allow_pickle=True)
            self.index += 1
            return data
        return None

class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(np.zeros([1, *data_loader.next_batch().shape[1:]], np.float32).nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        # print("batch.shape:", batch.shape)
        # print("batch.dtype:", batch.dtype)

        if batch is None:
            return None
        # batch index 관리 (self.batch_idx)
        if not hasattr(self, "batch_idx"):
            self.batch_idx = 0
        self.batch_idx += 1

        if self.batch_idx % 10 == 0:
            print(f"[Calibrator] batch {self.batch_idx} shape: {batch.shape}, dtype: {batch.dtype}")

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]
    # def get_batch(self, names):
    #     try:
    #         # Assume self.batches is a generator that provides batch data.
    #         data = next(self.batches)
    #         # Assume that self.device_input is a device buffer allocated by the constructor.
    #         cuda.memcpy_htod(self.device_input, data)
    #         return [int(self.device_input)]
    #     except StopIteration:
    #         # When we're out of batches, we return either [] or None.
    #         # This signals to TensorRT that there is no calibration data remaining.
    #         return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

