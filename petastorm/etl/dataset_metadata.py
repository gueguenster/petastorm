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

import json
import logging
import os
from concurrent import futures
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import Dict, Any, List

from packaging import version
from pyarrow import parquet as pq
from six.moves import cPickle as pickle

from petastorm import utils
from petastorm.etl.legacy import depickle_legacy_package_name_compatible
from petastorm.fs_utils import FilesystemResolver, get_filesystem_and_path_or_paths, get_dataset_path, path_exists
from petastorm.unischema import Unischema
from petastorm.utils import common_metadata_path, get_dataset_metadata_dict


@dataclass
class RowGroupIndices:
    fragment_index: int
    fragment_path: str
    row_group_id: int
    row_group_num_rows: int

    def to_dict(self) -> Dict[str, Any]:
        return {'fragment_index': self.fragment_index,
                'fragment_path': self.fragment_path,
                'row_group_id': self.row_group_id,
                'row_group_num_rows': self.row_group_num_rows}

logger = logging.getLogger(__name__)

ROW_GROUPS_PER_FILE_KEY = b'dataset-toolkit.num_row_groups_per_file.v1'
UNISCHEMA_KEY = b'dataset-toolkit.unischema.v1'


class PetastormMetadataError(Exception):
    """
    Error to specify when the petastorm metadata does not exist, does not contain the necessary information,
    or is corrupt/invalid.
    """


class PetastormMetadataGenerationError(Exception):
    """
    Error to specify when petastorm could not generate metadata properly.
    This error is usually accompanied with a message to try to regenerate dataset metadata.
    """


@contextmanager
def materialize_dataset(spark, dataset_url, schema, row_group_size_mb=None, use_summary_metadata=False,
                        filesystem_factory=None):
    """
    A Context Manager which handles all the initialization and finalization necessary
    to generate metadata for a petastorm dataset. This should be used around your
    spark logic to materialize a dataset (specifically the writing of parquet output).

    Note: Any rowgroup indexing should happen outside the materialize_dataset block

    Example:

    >>> spark = SparkSession.builder...
    >>> ds_url = 'hdfs:///path/to/my/dataset'
    >>> with materialize_dataset(spark, ds_url, MyUnischema, 64):
    >>>   spark.sparkContext.parallelize(range(0, 10)).
    >>>     ...
    >>>     .write.parquet(ds_url)
    >>> indexer = [SingleFieldIndexer(...)]
    >>> build_rowgroup_index(ds_url, spark.sparkContext, indexer)

    A user may provide their own recipe for creation of pyarrow filesystem object in ``filesystem_factory``
    argument (otherwise, petastorm will create a default one based on the url).

    The following example shows how a custom pyarrow HDFS filesystem, instantiated using ``libhdfs`` driver can be used
    during Petastorm dataset generation:

    >>> resolver=FilesystemResolver(dataset_url, spark.sparkContext._jsc.hadoopConfiguration(),
    >>>                             hdfs_driver='libhdfs')
    >>> with materialize_dataset(..., filesystem_factory=resolver.filesystem_factory()):
    >>>     ...


    :param spark: The spark session you are using
    :param dataset_url: The dataset url to output your dataset to (e.g. ``hdfs:///path/to/dataset``)
    :param schema: The :class:`petastorm.unischema.Unischema` definition of your dataset
    :param row_group_size_mb: The parquet row group size to use for your dataset
    :param use_summary_metadata: Whether to use the parquet summary metadata for row group indexing or a custom
      indexing method. The custom indexing method is more scalable for very large datasets.
    :param filesystem_factory: A filesystem factory function to be used when saving Petastorm specific metadata to the
      Parquet store.
    """
    spark_config = {}
    _init_spark(spark, spark_config, row_group_size_mb, use_summary_metadata)
    yield
    # After job completes, add the unischema metadata and check for the metadata summary file
    if filesystem_factory is None:
        resolver = FilesystemResolver(dataset_url, spark.sparkContext._jsc.hadoopConfiguration(),
                                      user=spark.sparkContext.sparkUser())
        filesystem_factory = resolver.filesystem_factory()
        dataset_path = resolver.get_dataset_path()
    else:
        dataset_path = get_dataset_path(urlparse(dataset_url))
    filesystem = filesystem_factory()

    dataset = pq.ParquetDataset(
        dataset_path,
        filesystem=filesystem)

    _generate_unischema_metadata(dataset, schema)
    _generate_num_row_groups_per_file(dataset)
    if use_summary_metadata:
        raise ValueError("This petastorm version does not support parquet summary metadata. Use use_summary_metadata=False.")

    # Reload the dataset to take into account the new metadata
    dataset = pq.ParquetDataset(
        dataset_path,
        filesystem=filesystem)
    try:
        # Try to load the row groups, if it fails that means the metadata was not generated properly
        load_row_groups(dataset)
    except PetastormMetadataError:
        raise PetastormMetadataGenerationError(
            'Could not find summary metadata file. The dataset will exist but you will need'
            ' to execute petastorm-generate-metadata.py before you can read your dataset '
            ' in order to generate the necessary metadata.'
            ' Try increasing spark driver memory next time and making sure you are'
            ' using parquet-mr >= 1.8.3')

    _cleanup_spark(spark, spark_config, row_group_size_mb)


def _init_spark(spark, current_spark_config, row_group_size_mb=None, use_summary_metadata=False):
    """
    Initializes spark and hdfs config with necessary options for petastorm datasets
    before running the spark job.
    """

    # It's important to keep pyspark import local because when placed at the top level it somehow messes up with
    # namedtuple serialization code and we end up getting UnischemaFields objects depickled without overriden __eq__
    # and __hash__ methods.
    import pyspark
    _PYSPARK_BEFORE_24 = version.parse(pyspark.__version__) < version.parse('2.4')

    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()

    # Store current values so we can restore them later
    current_spark_config['parquet.summary.metadata.level'] = \
        hadoop_config.get('parquet.summary.metadata.level')
    current_spark_config['parquet.enable.summary-metadata'] = \
        hadoop_config.get('parquet.enable.summary-metadata')
    current_spark_config['parquet.summary.metadata.propagate-errors'] = \
        hadoop_config.get('parquet.summary.metadata.propagate-errors')
    current_spark_config['parquet.block.size.row.check.min'] = \
        hadoop_config.get('parquet.block.size.row.check.min')
    current_spark_config['parquet.row-group.size.row.check.min'] = \
        hadoop_config.get('parquet.row-group.size.row.check.min')
    current_spark_config['parquet.block.size'] = \
        hadoop_config.get('parquet.block.size')

    if _PYSPARK_BEFORE_24:
        hadoop_config.setBoolean("parquet.enable.summary-metadata", use_summary_metadata)
    else:
        hadoop_config.set('parquet.summary.metadata.level', "ALL" if use_summary_metadata else "NONE")

    # Our atg fork includes https://github.com/apache/parquet-mr/pull/502 which creates this
    # option. This forces a job to fail if the summary metadata files cannot be created
    # instead of just having them fail to be created silently
    hadoop_config.setBoolean('parquet.summary.metadata.propagate-errors', True)
    # In our atg fork this config is called parquet.block.size.row.check.min however in newer
    # parquet versions it will be renamed to parquet.row-group.size.row.check.min
    # We use both for backwards compatibility
    hadoop_config.setInt('parquet.block.size.row.check.min', 3)
    hadoop_config.setInt('parquet.row-group.size.row.check.min', 3)
    if row_group_size_mb:
        hadoop_config.setInt('parquet.block.size', row_group_size_mb * 1024 * 1024)


def _cleanup_spark(spark, current_spark_config, row_group_size_mb=None):
    """
    Cleans up config changes performed in _init_spark
    """
    hadoop_config = spark.sparkContext._jsc.hadoopConfiguration()

    for key, val in current_spark_config.items():
        if val is not None:
            hadoop_config.set(key, val)
        else:
            hadoop_config.unset(key)


def _generate_unischema_metadata(dataset, schema):
    """
    Generates the serialized unischema and adds it to the dataset parquet metadata to be used upon reading.
    :param dataset: (ParquetDataset) Dataset to attach schema
    :param schema:  (Unischema) Schema to attach to dataset
    :return: None
    """
    # TODO(robbieg): Simply pickling unischema will break if the UnischemaField class is changed,
    #  or the codec classes are changed. We likely need something more robust.
    assert schema
    serialized_schema = pickle.dumps(schema)
    utils.add_to_dataset_metadata(dataset, UNISCHEMA_KEY, serialized_schema)


def _generate_num_row_groups_per_file(dataset):
    """
    Generates the metadata file containing the number of row groups in each file
    for the parquet dataset located at the dataset_url.
    :param dataset: :class:`pyarrow.parquet.ParquetDataset`
    :return: None, upon successful completion the metadata file will exist.
    """

    # Get the common prefix of all the base path in order to retrieve a relative path
    row_groups = [rg.to_dict() for rg in _row_groups(dataset)]
    num_row_groups_str = json.dumps(row_groups)
    # Add the dict for the number of row groups in each file to the parquet file metadata footer
    utils.add_to_dataset_metadata(dataset, ROW_GROUPS_PER_FILE_KEY, num_row_groups_str)

def _row_groups(dataset) -> list[RowGroupIndices]:
    # Force order of pieces. The order is not deterministic since it depends on multithreaded directory
    # listing implementation inside pyarrow. We stabilize order here, this way we get reproducable order
    # when pieces shuffling is off. This also enables implementing piece shuffling given a seed
    sorted_fragments = sorted(enumerate(dataset.fragments), key=lambda index_fragment: index_fragment[1].path)
    rowgroups = []
    for fragment_index, fragment in sorted_fragments:
        for row_group in fragment.row_groups:
            rowgroups.append(RowGroupIndices(fragment_index=fragment_index,
                                             fragment_path=fragment.path,
                                             row_group_id=row_group.id,
                                             row_group_num_rows=row_group.num_rows))
    return rowgroups

def load_row_groups(dataset: pq.ParquetDataset) -> list[RowGroupIndices]:
    """
    Load dataset row group pieces from metadata
    :param dataset: parquet dataset object.
    :return: list of RowGroupIndices
    """
    common_metadata_file_path = common_metadata_path(dataset, '_common_metadata')
    if path_exists(dataset.filesystem, common_metadata_file_path):
        dataset_metadata_dict = get_dataset_metadata_dict(common_metadata_file_path, filesystem=dataset.filesystem)
    else:
        dataset_metadata_dict = {}

    rowgroups: List[RowGroupIndices] = []
    if ROW_GROUPS_PER_FILE_KEY in dataset_metadata_dict:
        try:
            row_groups_list = json.loads(dataset_metadata_dict[ROW_GROUPS_PER_FILE_KEY].decode())
            rowgroups = [RowGroupIndices(**rg) for rg in row_groups_list]
        except Exception:
            # The row group data is not properly formatted, we recompute it.
            rowgroups = _row_groups(dataset)
            logger.warning(f"_common_metadata file contains an old metadata version for {ROW_GROUPS_PER_FILE_KEY}")
            logger.warning(f"We recompute the row_groups information which can take some time.")
    else:
        rowgroups = _row_groups(dataset)
    return rowgroups


# # This code has been copied (with small adjustments) from https://github.com/apache/arrow/pull/2223
# # Once that is merged and released this code can be deleted since we can use the open source
# # implementation.
# def _split_row_groups(dataset):
#     if not dataset.metadata or dataset.metadata.num_row_groups == 0:
#         raise NotImplementedError("split_row_groups is only implemented "
#                                   "if dataset has parquet summary files "
#                                   "with row group information")
#
#     # We make a dictionary of how many row groups are in each file in
#     # order to split them. The Parquet Metadata file stores paths as the
#     # relative path from the dataset base dir.
#     row_groups_per_file = dict()
#     for i in range(dataset.metadata.num_row_groups):
#         row_group = dataset.metadata.row_group(i)
#         path = row_group.column(0).file_path
#         row_groups_per_file[path] = row_groups_per_file.get(path, 0) + 1
#
#     base_path = os.path.normpath(os.path.dirname(dataset.metadata_path))
#     split_pieces = []
#     for piece in dataset.pieces:
#         # Since the pieces are absolute path, we get the
#         # relative path to the dataset base dir to fetch the
#         # number of row groups in the file
#         relative_path = os.path.relpath(piece.path, base_path)
#
#         # If the path is not in the metadata file, that means there are
#         # no row groups in that file and that file should be skipped
#         if relative_path not in row_groups_per_file:
#             continue
#
#         for row_group in range(row_groups_per_file[relative_path]):
#             split_piece = pq.ParquetDatasetPiece(piece.path, open_file_func=dataset.fs.open, row_group=row_group,
#                                                  partition_keys=piece.partition_keys)
#             split_pieces.append(split_piece)
#
#     return split_pieces
#
#
# def _split_piece(piece, fs_open):
#     metadata = piece.get_metadata()
#     return [pq.ParquetDatasetPiece(piece.path, open_file_func=fs_open,
#                                    row_group=row_group,
#                                    partition_keys=piece.partition_keys)
#             for row_group in range(metadata.num_row_groups)]
#
#
# def _split_row_groups_from_footers(dataset):
#     """Split the row groups by reading the footers of the parquet pieces"""
#
#     logger.info('Recovering rowgroup information for the entire dataset. This can take a long time for datasets with '
#                 'large number of files. If this dataset was generated by Petastorm '
#                 '(i.e. by using "with materialize_dataset(...)") and you still see this message, '
#                 'this indicates that the materialization did not finish successfully.')
#
#     thread_pool = futures.ThreadPoolExecutor()
#
#     futures_list = [thread_pool.submit(_split_piece, piece, dataset.fs.open) for piece in dataset.pieces]
#     result = [item for f in futures_list for item in f.result()]
#     thread_pool.shutdown()
#     return result

def get_schema(dataset: pq.ParquetDataset) -> Unischema:
    """Retrieves schema object stored as part of dataset methadata.

    :param dataset: an instance of :class:`pyarrow.parquet.ParquetDataset object`
    :return: A :class:`petastorm.unischema.Unischema` object
    """
    common_metadata_file_path = common_metadata_path(dataset, '_common_metadata')
    if path_exists(dataset.filesystem, common_metadata_file_path):
        dataset_metadata_dict = get_dataset_metadata_dict(common_metadata_file_path, filesystem=dataset.filesystem)
    else:
        dataset_metadata_dict = {}

    if not dataset_metadata_dict:
        raise PetastormMetadataError(
            'Could not find _common_metadata file. Use materialize_dataset(..) in'
            ' petastorm.etl.dataset_metadata.py to generate this file in your ETL code.'
            ' You can generate it on an existing dataset using petastorm-generate-metadata.py')

    # Read schema
    if UNISCHEMA_KEY not in dataset_metadata_dict:
        raise PetastormMetadataError(
            'Could not find the unischema in the dataset common metadata file.'
            ' Please provide or generate dataset with the unischema attached.'
            ' Common Metadata file might not be generated properly.'
            ' Make sure to use materialize_dataset(..) in petastorm.etl.dataset_metadata to'
            ' properly generate this file in your ETL code.'
            ' You can generate it on an existing dataset using petastorm-generate-metadata.py')
    ser_schema = dataset_metadata_dict[UNISCHEMA_KEY]
    # Since we have moved the unischema class around few times, unpickling old schemas will not work. In this case we
    # override the old import path to get backwards compatibility

    schema = depickle_legacy_package_name_compatible(ser_schema)

    return schema


def get_schema_from_dataset_url(dataset_url_or_urls, hdfs_driver='libhdfs3', storage_options=None, filesystem=None):
    """Returns a :class:`petastorm.unischema.Unischema` object loaded from a dataset specified by a url.

    :param dataset_url_or_urls: a url to a parquet directory or a url list (with the same scheme) to parquet files.
    :param hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
    :param storage_options: Dict of kwargs forwarded to ``fsspec`` to initialize the filesystem.
    :param fileystem: the ``pyarrow.FileSystem`` to use.
    :return: A :class:`petastorm.unischema.Unischema` object
    """
    fs, path_or_paths = get_filesystem_and_path_or_paths(dataset_url_or_urls, hdfs_driver,
                                                         storage_options=storage_options,
                                                         filesystem=filesystem)

    dataset = pq.ParquetDataset(path_or_paths, filesystem=fs)

    # Get a unischema stored in the dataset metadata.
    stored_schema = infer_or_load_unischema(dataset)

    return stored_schema


def infer_or_load_unischema(dataset):
    """Try to recover Unischema object stored by ``materialize_dataset`` function. If it can be loaded, infer
    Unischema from native Parquet schema"""
    try:
        return get_schema(dataset)
    except PetastormMetadataError:
        logger.info('Failed loading Unischema from metadata in %s. Assuming the dataset was not created with '
                    'Petastorm. Will try to construct from native Parquet schema.')
        return Unischema.from_arrow_schema(dataset)
