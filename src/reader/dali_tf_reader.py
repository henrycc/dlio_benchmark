"""
   Copyright (c) 2022, UChicago Argonne, LLC
   All Rights Reserved

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import math
import logging
from time import time

from src.utils.utility import utcnow, timeit
from src.common.enumerations import Shuffle
from src.reader.reader_handler import FormatReader
from nvidia.dali import pipeline_def
import tensorflow as tf
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf
import nvidia.dali.tfrecord as tfrec

class DaliTFReader(FormatReader):
    """
    Reader for TFRecord files.
    """
    def __init__(self, dataset_type):
        super().__init__(dataset_type)
        self.read_threads = self._args.read_threads
        self.computation_threads = self._args.computation_threads

    def read(self, epoch_number):
        """
        Sets up the tf data pipeline to read tf record files.
        Called once at the start of every epoch.
        Does not necessarily read in all the data at the start however.
        :param epoch_number:
        """
        # superclass function initializes the file list
        super().read(epoch_number)
        self._local_idx_list = [entry.rsplit('.', 1)[0] + ".idx" for entry in self._local_file_list]
        if self.read_threads==0:
            if self._args.my_rank==0:
                logging.warning(f"{utcnow()} `read_threads` is set to be 0 for tf.data loader. We change it to tf.data.AUTOTUNE")
            self.read_threads=tf.data.AUTOTUNE

        @pipeline_def(device_id=types.CPU_ONLY_DEVICE_ID, batch_size=self.batch_size, num_threads=self.read_threads)
        def pipeline_tfrecord():
            input = fn.readers.tfrecord(
                path=self._local_file_list,
                index_path=self._local_idx_list,
                features={
                        'image': tfrec.FixedLenFeature([], tfrec.string, ""),
                        'label': tfrec.FixedLenFeature([], tfrec.int64, -1)
                },
                use_o_direct=True,
                dont_use_mmap=True
            )
            label = input['label']
            image = input['image']
            return image, label

        pipe = pipeline_tfrecord()
        dataset = dali_tf.DALIDataset(
            pipeline=pipe,
            batch_size=self.batch_size,
            num_threads=self.read_threads,
            prefetch_queue_depth=32,
            cpu_prefetch_queue_depth=32,
            output_shapes=None,
            device_id=types.CPU_ONLY_DEVICE_ID,
            output_dtypes=(tf.uint8, tf.int64)
        )

        if self.sample_shuffle != Shuffle.OFF:
            if self.sample_shuffle == Shuffle.SEED:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size,
                                          seed=self.seed)
            else:
                dataset = dataset.shuffle(buffer_size=self.shuffle_size)

        self._dataset = dataset.batch(self.batch_size, drop_remainder=True)


    def next(self):
        """
        Provides the iterator over tfrecord data pipeline.
        :return: data to be processed by the training step.
        """
        super().next()

        # In tf, we can't get the length of the dataset easily so we calculate it
        if self._debug:
            total = math.floor(self.num_samples*len(self._file_list)/self.batch_size/self.comm_size)
            logging.debug(f"{utcnow()} Rank {self.my_rank} should read {total} batches")

        # The previous version crashed when all workers could not generate the same amount of batches
        # Using the inbuilt tensorflow dataset iteration seems to work fine, was there an advantage of doing it the old way?
        for batch in self._dataset:
            yield batch

    def finalize(self):
        pass
