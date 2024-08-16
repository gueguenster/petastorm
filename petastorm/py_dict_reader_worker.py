#  Copyright (c) 2017-2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import hashlib
import threading
from collections.abc import Iterable
from typing import Tuple, Set

import numpy as np
from pyarrow import parquet as pq
from pyarrow.dataset import ParquetFileFragment, get_partition_keys

from petastorm import utils
from petastorm.cache import NullCache
from petastorm.etl.dataset_metadata import RowGroupIndices
from petastorm.workers_pool import EmptyResultError
from petastorm.workers_pool.worker_base import WorkerBase


def _merge_two_dicts(a, b):
    """Merges two dictionaries together. If the same key is present in both input dictionaries, the value from 'b'
    dominates."""
    result = a.copy()
    result.update(b)
    return result


def _apply_transform_spec(all_rows, transform_spec):
    """Applies transform_spec.func to all rows in all_rows (if func is specified). Removes all fields as specified by
    transform_spec.removed_fields, unless the fields were already deleted by transform_spec.func.

    All modifications are performed in-place.
    """
    if transform_spec.func:
        all_rows = [transform_spec.func(row) for row in all_rows]

    for field_to_remove in transform_spec.removed_fields:
        for row in all_rows:
            if field_to_remove in row:
                del row[field_to_remove]

    return all_rows


def _select_cols(a_dict, keys):
    """Filters out entries in a dictionary that have a key which is not part of 'keys' argument. `a_dict` is not
    modified and a new dictionary is returned."""
    if keys == list(a_dict.keys()):
        return a_dict
    else:
        return {field_name: a_dict[field_name] for field_name in keys}


class PyDictReaderWorkerResultsQueueReader(object):
    def __init__(self):
        self._result_buffer_lock = threading.Lock()
        self._result_buffer = []

    @property
    def batched_output(self):
        return False

    def read_next(self, workers_pool, schema, ngram):
        try:
            # We are receiving decoded rows from the worker in chunks. We store the list internally
            # and return a single item upon each consequent call to __next__
            with self._result_buffer_lock:
                if not self._result_buffer:
                    # Reverse order, so we can pop from the end of the list in O(1) while maintaining
                    # order the items are returned from the worker
                    list_of_rows = list(reversed(workers_pool.get_results()))

                    if ngram:
                        for ngram_row in list_of_rows:
                            for timestamp in ngram_row.keys():
                                row = ngram_row[timestamp]
                                schema_at_timestamp = ngram.get_schema_at_timestep(schema, timestamp)

                                ngram_row[timestamp] = schema_at_timestamp.make_namedtuple(**row)
                        self._result_buffer = list_of_rows
                    else:
                        self._result_buffer = [schema.make_namedtuple(**row) for row in list_of_rows]

                return self._result_buffer.pop()

        except EmptyResultError:
            raise StopIteration


class PyDictReaderWorker(WorkerBase):
    def __init__(self, worker_id, publish_func, args):
        super(PyDictReaderWorker, self).__init__(worker_id, publish_func, args)

        self._filesystem = args[0]
        self._dataset_path = args[1]
        self._schema = args[2]
        self._ngram = args[3]
        self._split_pieces = args[4]
        self._local_cache = args[5]
        self._transform_spec = args[6]
        self._arrow_filters = args[8]
        self._shuffle_rows = args[9]
        self._shuffle_seed = args[10]

        # We create datasets lazily in the first invocation of 'def process'. This speeds up startup time since
        # all Worker constructors are serialized
        self._dataset = None
        self._partition_columns = None

    @staticmethod
    def new_results_queue_reader():
        return PyDictReaderWorkerResultsQueueReader()

    # pylint: disable=arguments-differ
    def process(self, piece_index: int, worker_predicate, shuffle_row_drop_partition: Tuple[int, int]):
        """Main worker function. Loads and returns all rows matching the predicate from a rowgroup

        Looks up the requested piece (a single row-group in a parquet file). If a predicate is specified,
        columns needed by the predicate are loaded first. If no rows in the rowgroup matches the predicate criteria
        the rest of the columns are not loaded.

        :param piece_index:
        :param shuffle_row_drop_partition: A tuple 2 of the current row drop partition and the total number
            of partitions.
        :return:
        """

        if not self._dataset:
            self._dataset = pq.ParquetDataset(
                self._dataset_path,
                filesystem=self._filesystem,
                filters=self._arrow_filters)

        row_group_indices: RowGroupIndices = self._split_pieces[piece_index]

        # Create pyarrow file system
        parquet_file = self._dataset.fragments[row_group_indices.fragment_index]

        if not isinstance(self._local_cache, NullCache):
            if worker_predicate:
                raise RuntimeError('Local cache is not supported together with predicates, '
                                   'unless the dataset is partitioned by the column the predicate operates on.')
            if shuffle_row_drop_partition[1] != 1:
                raise RuntimeError('Local cache is not supported together with shuffle_row_drop_partitions > 1')

        if worker_predicate:
            all_cols = self._load_rows_with_predicate(pq_file=parquet_file,
                                                      piece=row_group_indices,
                                                      worker_predicate=worker_predicate,
                                                      shuffle_row_drop_partition=shuffle_row_drop_partition)
        else:
            # Using hash of the dataset path with the relative path in order to:
            #  1. Make sure if a common cache serves multiple processes (e.g. redis), we don't have conflicts
            #  2. Dataset path is hashed, to make sure we don't create too long keys, which maybe incompatible with
            #     some cache implementations
            #  3. Still leave relative path and the piece_index in plain text to make it easier to debug
            # self._dataset_path could be a list of urls or a string.
            _dataset_path_for_hash = "_".join(self._dataset_path) if isinstance(self._dataset_path,
                                                                                Iterable) else self._dataset_path
            cache_key = '{}:{}:{}'.format(hashlib.md5(_dataset_path_for_hash.encode('utf-8')).hexdigest(),
                                          row_group_indices.fragment_path, piece_index)
            all_cols = self._local_cache.get(cache_key,
                                             lambda: self._load_rows(pq_file=parquet_file,
                                                                     piece=row_group_indices,
                                                                     shuffle_row_drop_range=shuffle_row_drop_partition))

        if self._ngram:
            all_cols = self._ngram.form_ngram(data=all_cols, schema=self._schema)

        if all_cols:
            self.publish_func(all_cols)

    def _load_rows(self, pq_file: ParquetFileFragment, piece: RowGroupIndices, shuffle_row_drop_range: Tuple[int, int]):
        """Loads all rows from a piece"""
        column_names = set(field.name for field in self._schema.fields.values())

        all_rows = self._read_with_shuffle_row_drop(piece=piece,
                                                    pq_file=pq_file,
                                                    column_names=column_names,
                                                    shuffle_row_drop_partition=shuffle_row_drop_range)

        all_rows = [utils.decode_row(row, self._schema) for row in all_rows]

        if self._transform_spec:
            all_rows = _apply_transform_spec(all_rows, self._transform_spec)

        return all_rows

    def _load_rows_with_predicate(self, pq_file: ParquetFileFragment, piece: RowGroupIndices, worker_predicate, shuffle_row_drop_partition: Tuple[int, int]):
        """Loads all rows that match a predicate from a piece"""

        # 1. Read all columns needed by predicate and decode
        # 2. Apply the predicate. If nothing matches, exit early
        # 3. Read the remaining columns and decode
        # 4. Combine with columns already decoded for the predicate.

        # Split all column names into ones that are needed by predicate and the rest.
        predicate_column_names = set(worker_predicate.get_fields())

        if not predicate_column_names:
            raise ValueError('At least one field name must be returned by predicate\'s get_field() method')

        all_schema_names = set(field.name for field in self._schema.fields.values())

        invalid_column_names = predicate_column_names - all_schema_names
        if invalid_column_names:
            raise ValueError('At least some column names requested by the predicate ({}) '
                             'are not valid schema names: ({})'.format(', '.join(invalid_column_names),
                                                                       ', '.join(all_schema_names)))

        other_column_names = all_schema_names - predicate_column_names

        # Read columns needed for the predicate
        predicate_rows = self._read_with_shuffle_row_drop(piece=piece,
                                                          pq_file=pq_file,
                                                          column_names=predicate_column_names,
                                                          shuffle_row_drop_partition=shuffle_row_drop_partition)

        # Decode values
        decoded_predicate_rows = [
            utils.decode_row(_select_cols(row, predicate_column_names), self._schema)
            for row in predicate_rows]

        # Use the predicate to filter
        match_predicate_mask = [worker_predicate.do_include(row) for row in decoded_predicate_rows]

        # Don't have anything left after filtering? Exit early.
        if not any(match_predicate_mask):
            return []

        # Remove rows that were filtered out by the predicate
        filtered_decoded_predicate_rows = [row for i, row in enumerate(decoded_predicate_rows) if
                                           match_predicate_mask[i]]

        if other_column_names:
            # Read remaining columns
            other_rows = self._read_with_shuffle_row_drop(piece=piece,
                                                          pq_file=pq_file,
                                                          column_names=other_column_names,
                                                          shuffle_row_drop_partition=shuffle_row_drop_partition)

            # Remove rows that were filtered out by the predicate
            filtered_other_rows = [row for i, row in enumerate(other_rows) if match_predicate_mask[i]]

            # Decode remaining columns
            decoded_other_rows = [utils.decode_row(row, self._schema) for row in filtered_other_rows]

            # Merge predicate needed columns with the remaining
            all_cols = [_merge_two_dicts(a, b) for a, b in zip(decoded_other_rows, filtered_decoded_predicate_rows)]
            result = all_cols
        else:
            result = filtered_decoded_predicate_rows

        if self._transform_spec:
            result = _apply_transform_spec(result, self._transform_spec)

        return result

    def _read_with_shuffle_row_drop(self, piece: RowGroupIndices,
                                    pq_file: ParquetFileFragment,
                                    column_names: Set[str],
                                    shuffle_row_drop_partition: Tuple[int, int]):

        if not self._ngram:
            # If not ngram we can subset.
            # If ngram we have to read the full fragment because we want to access sub-sequences
            pq_file = pq_file.subset(row_group_ids=[piece.row_group_id])

        partition_dict = get_partition_keys(pq_file.partition_expression)
        partition_columns = set(partition_dict.keys())
        non_partitioned_columns = list(column_names - partition_columns)

        data_frame = pq_file.to_table(columns=non_partitioned_columns, filter=self._dataset._filter_expression).to_pandas(integer_object_nulls=True)

        if self._shuffle_rows:
            data_frame = data_frame.sample(frac=1, random_state=self._shuffle_seed)

        num_rows = len(data_frame)
        num_partitions = shuffle_row_drop_partition[1]
        this_partition = shuffle_row_drop_partition[0]

        if num_rows > 0:
            partition_indexes = np.floor(np.arange(num_rows) / (float(num_rows) / min(num_rows, num_partitions)))
        else:
            partition_indexes = np.floor(np.arange(num_rows))

        if self._ngram:
            # If we have an ngram we need to take elements from the next partition to build the sequence
            next_partition_indexes = np.where(partition_indexes >= this_partition + 1)
            if next_partition_indexes[0].size:
                next_partition_to_add = next_partition_indexes[0][0:self._ngram.length - 1]
                partition_indexes[next_partition_to_add] = this_partition

        selected_dataframe = data_frame.loc[partition_indexes == this_partition]

        # Add back partition data, which is constant for this data fragment
        for partition_key, partition_value in partition_dict.items():
            if column_names is not None and partition_key in column_names:
                selected_dataframe[partition_key] = partition_value

        return selected_dataframe.to_dict('records')

