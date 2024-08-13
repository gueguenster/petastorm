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

import logging
import os
from decimal import Decimal
from multiprocessing import Pool

import numpy as np
import pyarrow
from future.utils import raise_with_traceback
from pyarrow.fs import LocalFileSystem, FileType
from pyarrow.parquet import ParquetDataset, ParquetFile

logger = logging.getLogger(__name__)


def run_in_subprocess(func, *args, **kwargs):
    """
    Run some code in a separate process and return the result. Once the code is done, terminate the process.
    This prevents a memory leak in the other process from affecting the current process.

    Gotcha: func must be a functioned defined at the top level of the module.
    :param kwargs: dict
    :param args: list
    :param func:
    :return:
    """
    pool = Pool(1)
    result = pool.apply(func, args=args, kwds=kwargs)

    # Probably not strictly necessary since terminate is called on GC, but it's not guaranteed when the pool will get
    # GC'd.
    pool.terminate()
    return result


class DecodeFieldError(RuntimeError):
    pass


def decode_row(row, schema):
    """
    Decode dataset row according to coding spec from unischema object

    If a codec is set, we use codec.decode() to produce decoded value.

    For scalar fields, the codec maybe set to `None`. In that case:
     - If the numpy_dtype is a numpy scalar or a Decimal, cast the 'encoded' value before returning.
     - In any other case, return the value 'as is'.

    :param row: dictionary with encodded values
    :param schema: unischema object
    :return:
    """
    decoded_row = dict()
    for field_name_unicode, _ in row.items():
        field_name = str(field_name_unicode)
        if field_name in schema.fields:
            try:
                if row[field_name] is not None:
                    field = schema.fields[field_name]
                    codec = schema.fields[field_name].codec
                    if codec:
                        decoded_row[field_name] = codec.decode(field, row[field_name])
                    elif field.numpy_dtype and issubclass(field.numpy_dtype, (np.generic, Decimal)):
                        decoded_row[field_name] = field.numpy_dtype(row[field_name])
                    else:
                        decoded_row[field_name] = row[field_name]
                else:
                    decoded_row[field_name] = None
            except Exception:  # pylint: disable=broad-except
                raise_with_traceback(DecodeFieldError('Decoding field "{}" failed'.format(field_name)))

    return decoded_row


def path_exists(fs, path):
    file_info = fs.get_file_info(path)
    return not file_info.type == FileType.NotFound


def common_metadata_path(dataset: ParquetDataset, suffix: str = "_common_metadata") -> str:
    """
    Returns the common metadata path for the dataset.
    """
    if isinstance(dataset.files, list) and len(dataset.files) > 0:
        base_path = os.path.dirname(dataset.files[0])
    else:
        raise Exception(f"{dataset} does not contain any files.")

    file_path = base_path.rstrip('/') + '/' + suffix
    return file_path


def get_dataset_metadata_dict(dataset_metadata_path: str, filesystem=None) -> dict:
    """
    Returns the dictionary from the common dataset metadata path.
    """
    pqf = ParquetFile(dataset_metadata_path, filesystem=filesystem)
    return pqf.schema.to_arrow_schema().metadata


def add_to_dataset_metadata(dataset: ParquetDataset, key: bytes, value: bytes):
    """
    Adds a key and value to the parquet metadata file of a parquet dataset.
    :param dataset: (ParquetDataset) parquet dataset
    :param key:     (bytes) key of metadata entry
    :param value:   (bytes) value of metadata
    """

    common_metadata_file_path = common_metadata_path(dataset=dataset, suffix='_common_metadata')
    common_metadata_file_crc_path = common_metadata_path(dataset=dataset, suffix='._common_metadata.crc')
    metadata_file_path = common_metadata_path(dataset=dataset, suffix='_metadata')

    # If the metadata file already exists, add to it.
    # Otherwise fetch the schema from one of the existing parquet files in the dataset
    fs = dataset.filesystem
    metadata_dict = dataset.schema.metadata
    if path_exists(fs, common_metadata_file_path):
        metadata_dict = get_dataset_metadata_dict(common_metadata_file_path, fs)
    elif path_exists(fs, metadata_file_path):
        metadata_dict = get_dataset_metadata_dict(metadata_file_path, fs)

    # base_schema.metadata may be None, e.g.
    metadata_dict = metadata_dict or dict()
    metadata_dict[key] = value
    new_schema = dataset.schema.with_metadata(metadata_dict)
    pyarrow.parquet.write_metadata(new_schema, common_metadata_file_path, filesystem=fs)

    # We have just modified _common_metadata file, but the filesystem implementation used by pyarrow does not
    # update the .crc value. We better delete the .crc to make sure there is no mismatch between _common_metadata
    # content and the checksum.
    if path_exists(fs, common_metadata_file_crc_path):
        fs.delete_file(common_metadata_file_crc_path)
