# Copyright (c) 2013-2020, Freja Nordsiek
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import collections
import copy
import datetime
import itertools
import math
import os
import os.path
import pathlib
import posixpath
import random
import string
import tempfile

import numpy as np

import hdf5storage

from asserts import assert_equal, assert_equal_none_format, \
    assert_equal_matlab_format
from make_randoms import random_numpy_scalar, random_numpy_shape, \
    random_numpy, random_name, max_array_axis_length, \
    max_structured_ndarray_axis_length, random_structured_numpy_array, \
    random_str_ascii, max_dict_key_length, \
    random_list, max_list_length, random_dict, \
    random_str_some_unicode, random_int, random_float, \
    max_string_length, random_bytes, random_slice, random_range, \
    random_chainmap, random_fraction, random_datetime_timezone


random.seed()


class TestPythonMatlabFormat(object):
    # Test for the ability to write python types to an HDF5 file that
    # type information and matlab information are stored in, and then
    # read it back and have it be the same.
    def __init__(self):
        self.options = hdf5storage.Options()

        # Need a list of the supported numeric dtypes to test, excluding
        # those not supported by MATLAB. 'S' and 'U' dtype chars have to
        # be used for the bare byte and unicode string dtypes since the
        # dtype strings (but not chars) are not the same in Python 2 and
        # 3.
        self.dtypes = ['bool', 'uint8', 'uint16', 'uint32', 'uint64',
                       'int8', 'int16', 'int32', 'int64',
                       'float32', 'float64', 'complex64', 'complex128',
                       'S', 'U']

        # Need a list of dict-like types, which will depend on Python
        # version.
        self.dict_like = ['dict', 'OrderedDict', 'Counter']

    def write_readback(self, data, name, options, read_options=None):
        # Randomly convert the name to other path types.
        path_choices = (str, bytes,
                        pathlib.PurePath,
                        pathlib.PurePosixPath,
                        pathlib.PureWindowsPath,
                        pathlib.Path)
        name_type_w = random.choice(path_choices)
        name_type_r = random.choice(path_choices)
        # Name to write with.
        if name_type_w == bytes:
            name_w = name.encode('utf-8')
        elif name_type_w in (pathlib.PurePath, pathlib.PurePosixPath,
                             pathlib.PosixPath):
            name_w = name_type_w(name)
        elif name_type_w != str:
            name_w = name_type_w(name[posixpath.isabs(name):])
        else:
            name_w = name
        # Name to read with.
        if name_type_r == bytes:
            name_r = name.encode('utf-8')
        elif name_type_r in (pathlib.PurePath, pathlib.PurePosixPath,
                             pathlib.PosixPath):
            name_r = name_type_r(name)
        elif name_type_r != str:
            name_r = name_type_r(name[posixpath.isabs(name):])
        else:
            name_r = name
        # Write the data to the proper file with the given name, read it
        # back, and return the result. The file needs to be deleted
        # after to keep junk from building up. Different options can be
        # used for reading the data back.
        f = None
        try:
            f = tempfile.mkstemp()
            os.close(f[0])
            filename = f[1]
            hdf5storage.write(data, path=name_w, filename=filename,
                              options=options)
            out = hdf5storage.read(path=name_r, filename=filename,
                                   options=read_options)
        except:
            raise
        finally:
            if f is not None:
                os.remove(f[1])
        return out

    def assert_equal(self, a, b):
        assert_equal(a, b, self.options)

    def check_numpy_scalar(self, dtype):
        # Makes a random numpy scalar of the given type, writes it and
        # reads it back, and then compares it.
        data = random_numpy_scalar(dtype)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_array(self, dtype, dimensions):
        # Makes a random numpy array of the given type, writes it and
        # reads it back, and then compares it.
        shape = random_numpy_shape(dimensions,
                                   max_array_axis_length)
        data = random_numpy(shape, dtype)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_empty(self, dtype):
        # Makes an empty numpy array of the given type, writes it and
        # reads it back, and then compares it.
        data = np.array([], dtype)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_stringlike_empty(self, dtype, num_chars):
        # Makes an empty stringlike numpy array of the given type and
        # size, writes it and reads it back, and then compares it.
        data = np.array([], dtype + str(num_chars))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)
    
    def check_numpy_structured_array(self, dimensions):
        # Makes a random structured ndarray of the given type, writes it
        # and reads it back, and then compares it.
        shape = random_numpy_shape(dimensions, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)
    
    def check_numpy_structured_array_empty(self, dimensions):
        # Makes a random structured ndarray of the given type, writes it
        # and reads it back, and then compares it.
        shape = random_numpy_shape(dimensions, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape, (1, 0))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_structured_array_field_special_char(self, ch,
                                                        leading=False):
        # Makes a random 1d structured ndarray with the character
        # in one field, writes it and reads it back, and then compares
        # it.
        field_names = [random_str_ascii(max_dict_key_length)
                       for i in range(2)]
        if leading:
            field_names[1] = ch + field_names[1]
        else:
            field_names[1] = field_names[1][0] + ch + field_names[1][1:]
        field_names[1] = field_names[1][0] + ch + field_names[1][1:]
        shape = random_numpy_shape(1, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape, names=field_names)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_matrix(self, dtype):
        # Makes a random numpy array of the given type, converts it to
        # a matrix, writes it and reads it back, and then compares it.
        shape = random_numpy_shape(2, max_array_axis_length)
        data = np.matrix(random_numpy(shape, dtype))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_recarray(self, dimensions):
        # Makes a random structured ndarray of the given type, converts
        # it to a recarray, writes it and reads it back, and then
        # compares it.
        shape = random_numpy_shape(dimensions, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape).view(np.recarray).copy()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_recarray_empty(self, dimensions):
        # Makes a random structured ndarray of the given type, converts
        # it to a recarray, writes it and reads it back, and then
        # compares it.
        shape = random_numpy_shape(dimensions, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape, (1, 0)).view(np.recarray).copy()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_recarray_field_special_char(self, ch,
                                                leading=False):
        # Makes a random 1d structured ndarray with the character
        # in one field, converts it to a recarray, writes it and reads
        # it back, and then compares it.
        field_names = [random_str_ascii(max_dict_key_length)
                       for i in range(2)]
        if leading:
            field_names[1] = ch + field_names[1]
        else:
            field_names[1] = field_names[1][0] + ch + field_names[1][1:]
        shape = random_numpy_shape(1, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape, names=field_names).view(np.recarray).copy()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_chararray(self, dimensions):
        # Makes a random numpy array of bytes, converts it to a
        # chararray, writes it and reads it back, and then compares it.
        shape = random_numpy_shape(dimensions,
                                   max_array_axis_length)
        data = random_numpy(shape, 'S').view(np.chararray).copy()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_chararray_empty(self, num_chars):
        # Makes an empty numpy array of bytes of the given number of
        # characters, converts it to a chararray, writes it and reads it
        # back, and then compares it.
        data = np.array([], 'S' + str(num_chars)).view(np.chararray).copy()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_numpy_sized_dtype_nested_0(self, zero_shaped):
        dtypes = ('uint8', 'uint16', 'uint32', 'uint64',
                  'int8', 'int16', 'int32', 'int64',
                  'float32', 'float64',
                  'complex64', 'complex128')
        for i in range(10):
            dt = (random.choice(dtypes), (2, 2 * zero_shaped))
            data = np.zeros((2, ), dtype=dt)
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def check_numpy_sized_dtype_nested_1(self, zero_shaped):
        dtypes = ('uint8', 'uint16', 'uint32', 'uint64',
                  'int8', 'int16', 'int32', 'int64',
                  'float32', 'float64',
                  'complex64', 'complex128')
        for i in range(10):
            dt = [('a', random.choice(dtypes), (1, 2)),
                  ('b', random.choice(dtypes), (1, 1, 4 * zero_shaped)),
                  ('c',
                   [('a', random.choice(dtypes)),
                    ('b', random.choice(dtypes), (1, 2))])]
            data = np.zeros((random.randrange(1, 4), ), dtype=dt)
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def check_numpy_sized_dtype_nested_2(self, zero_shaped):
        dtypes = ('uint8', 'uint16', 'uint32', 'uint64',
                  'int8', 'int16', 'int32', 'int64',
                  'float32', 'float64',
                  'complex64', 'complex128')
        for i in range(10):
            dt = [('a', random.choice(dtypes), (1, 3)),
                  ('b',
                   [('a', random.choice(dtypes), (2, )),
                    ('b', random.choice(dtypes), (1, 2, 1))]),
                  ('c',
                   [('a', random.choice(dtypes),
                     (3 * zero_shaped, 1)),
                    ('b', random.choice(dtypes), (2, ))], (2, 1))]
            data = np.zeros((2, ), dtype=dt)
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def check_numpy_sized_dtype_nested_3(self, zero_shaped):
        dtypes = ('uint8', 'uint16', 'uint32', 'uint64',
                  'int8', 'int16', 'int32', 'int64',
                  'float32', 'float64',
                  'complex64', 'complex128')
        for i in range(10):
            dt = [('a', random.choice(dtypes), (3, 2)),
                  ('b',
                   [('a',
                     [('a', random.choice(dtypes))],
                     (2, 2)),
                    ('b', random.choice(dtypes), (1, 2))]),
                  ('c',
                   [('a',
                     [('a', random.choice(dtypes),
                       (2, 1, zero_shaped * 2))]),
                    ('b', random.choice(dtypes))])]
            data = np.zeros((1, ), dtype=dt)
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def check_python_collection(self, tp, same_dims):
        # Makes a random collection of the specified type, writes it and
        # reads it back, and then compares it.
        if tp in (set, frozenset):
            data = tp(random_list(max_list_length,
                      python_or_numpy='python'))
        else:
            if same_dims == 'same-dims':
                shape = random_numpy_shape(random.randrange(2, 4),
                                           random.randrange(1, 4))
                dtypes = ('uint8', 'uint16', 'uint32', 'uint64',
                          'int8', 'int16', 'int32', 'int64',
                          'float32', 'float64',
                          'complex64', 'complex128')
                data = tp([random_numpy(shape, random.choice(dtypes),
                                        allow_nan=True)
                           for i in range(random.randrange(2, 7))])

            elif same_dims == 'diff-dims':
                data = tp(random_list(max_list_length,
                                      python_or_numpy='numpy'))
            else:
                raise ValueError('invalid value of same_dims')
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dict_like(self, tp):
        data = random_dict(tp)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dict_like_other_type_key(self, tp, other_tp):
        data = random_dict(tp)

        key_gen = random_str_some_unicode(max_dict_key_length)
        if other_tp == 'numpy.bytes_':
            key = np.bytes_(key_gen.encode('UTF-8'))
        elif other_tp == 'numpy.unicode_':
            key = np.unicode_(key_gen)
        elif other_tp == 'bytes':
            key = key_gen.encode('UTF-8')
        elif other_tp == 'int':
            key = random_int()
        elif other_tp == 'float':
            key = random_float()

        data[key] = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dict_like_key_null_character(self, tp):
        data = random_dict(tp)
        ch = '\x00'
        key = ch.join([random_str_ascii(max_dict_key_length)
                      for i in range(2)])
        data[key] = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dict_like_key_forward_slash(self, tp):
        data = random_dict(tp)
        ch = '/'
        key = ch.join([random_str_ascii(max_dict_key_length)
                      for i in range(2)])
        data[key] = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dict_like_key_back_slash(self, tp):
        data = random_dict(tp)
        ch = '\\'
        key = ch.join([random_str_ascii(max_dict_key_length)
                      for i in range(2)])
        data[key] = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dict_like_key_leading_periods(self, tp):
        data = random_dict(tp)
        prefix = '.' * random.randint(1, 10)
        key = prefix + random_str_ascii(max_dict_key_length)
        data[key] = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def check_dtype(self, dt):
        out = self.write_readback(dt, random_name(), self.options)
        self.assert_equal(out, dt)

    def test_None(self):
        data = None
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_Ellipsis(self):
        data = Ellipsis
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_NotImplemented(self):
        data = NotImplemented
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bool_True(self):
        data = True
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bool_False(self):
        data = False
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_int_needs_32_bits(self):
        data = random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_int_needs_64_bits(self):
        data = (2**32) * random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_int_or_long_too_big(self):
        data = 2**64 * random_int()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float(self):
        data = random_float()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float_inf(self):
        data = float(np.inf)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float_ninf(self):
        data = float(-np.inf)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_float_nan(self):
        data = float(np.nan)
        out = self.write_readback(data, random_name(),
                                  self.options)
        assert math.isnan(out)

    def test_complex(self):
        data = random_float() + 1j*random_float()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_complex_real_nan(self):
        data = complex(np.nan, random_float())
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_complex_imaginary_nan(self):
        data = complex(random_float(), np.nan)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_ascii(self):
        data = random_str_ascii(random.randint(1,
                                max_string_length))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_ascii_encoded_utf8(self):
        ltrs = string.ascii_letters + string.digits
        data = 'a'
        while all([(c in ltrs) for c in data]):
            data = random_str_some_unicode(random.randint(1, \
                max_string_length))
        data = data.encode('utf-8')
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_with_null(self):
        strs = [random_str_ascii(
                random.randint(1, max_string_length))
                for i in range(2)]
        data = '\x00'.join(strs)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_unicode(self):
        data = random_str_some_unicode(random.randint(1,
                                       max_string_length))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_str_empty(self):
        data = ''
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytes(self):
        data = random_bytes(random.randint(1,
                            max_string_length))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytes_empty(self):
        data = b''
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytes_with_null(self):
        strs = [random_bytes(
                random.randint(1, max_string_length))
                for i in range(2)]
        data = b'\x00'.join(strs)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytearray(self):
        data = bytearray(random_bytes(random.randint(1,
                         max_string_length)))
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_bytearray_empty(self):
        data = bytearray(b'')
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_numpy_scalar(self):
        for dt in self.dtypes:
            yield self.check_numpy_scalar, dt

    def test_numpy_array_1d(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_array, dt, 1

    def test_numpy_array_2d(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_array, dt, 2

    def test_numpy_array_3d(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_array, dt, 3

    def test_numpy_matrix(self):
        dts = copy.deepcopy(self.dtypes)
        dts.append('object')
        for dt in dts:
            yield self.check_numpy_matrix, dt

    def test_numpy_empty(self):
        for dt in self.dtypes:
            yield self.check_numpy_empty, dt

    def test_numpy_stringlike_empty(self):
        dts = ['S', 'U']
        for dt in dts:
            for n in range(1,10):
                yield self.check_numpy_stringlike_empty, dt, n
    
    def test_numpy_structured_array(self):
        for i in range(1, 4):
            yield self.check_numpy_structured_array, i
    
    def test_numpy_structured_array_empty(self):
        for i in range(1, 4):
            yield self.check_numpy_structured_array_empty, i

    def test_numpy_structured_array_unicode_fields(self):
        # Makes a random 1d structured ndarray with non-ascii characters
        # in its fields, writes it and reads it back, and then compares
        # it.
        shape = random_numpy_shape(1, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape, \
            nonascii_fields=True).view(np.recarray).copy()
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_numpy_structured_array_field_null_character(self):
        self.check_numpy_structured_array_field_special_char('\x00')

    def test_numpy_structured_array_field_forward_slash(self):
        self.check_numpy_structured_array_field_special_char('/')

    def test_numpy_structured_array_field_back_slash(self):
        self.check_numpy_structured_array_field_special_char('\\')

    def test_numpy_structured_array_field_leading_periods(self):
        s = '.' * random.randint(1, 10)
        self.check_numpy_structured_array_field_special_char( \
            s, leading=True)

    def test_numpy_recarray(self):
        for i in range(1, 4):
            yield self.check_numpy_recarray, i

    def test_numpy_recarray_empty(self):
        for i in range(1, 4):
            yield self.check_numpy_recarray_empty, i

    def test_numpy_recarray_unicode_fields(self):
        # Makes a random 1d structured ndarray with non-ascii characters
        # in its fields, converts it to a recarray, writes it and reads
        # it back, and then compares it.
        shape = random_numpy_shape(1, \
            max_structured_ndarray_axis_length)
        data = random_structured_numpy_array(shape,
                                             nonascii_fields=True)
        out = self.write_readback(data, random_name(),
                                  self.options)
        self.assert_equal(out, data)

    def test_numpy_recarray_field_null_character(self):
        self.check_numpy_recarray_field_special_char('\x00')

    def test_numpy_recarray_field_forward_slash(self):
        self.check_numpy_recarray_field_special_char('/')

    def test_numpy_recarray_field_back_slash(self):
        self.check_numpy_recarray_field_special_char('\\')

    def test_numpy_recarray_field_leading_periods(self):
        s = '.' * random.randint(1, 10)
        self.check_numpy_recarray_field_special_char(s, leading=True)

    def test_numpy_chararray(self):
        dims = range(1, 4)
        for dim in dims:
            yield self.check_numpy_chararray, dim

    def test_numpy_chararray_empty(self):
        for n in range(1, 10):
            yield self.check_numpy_chararray_empty, n

    def test_numpy_sized_dtype_nested_0(self):
        for zero_shaped in (False, True):
            yield self.check_numpy_sized_dtype_nested_0, zero_shaped

    def test_numpy_sized_dtype_nested_1(self):
        for zero_shaped in (False, True):
            yield self.check_numpy_sized_dtype_nested_1, zero_shaped

    def test_numpy_sized_dtype_nested_2(self):
        for zero_shaped in (False, True):
            yield self.check_numpy_sized_dtype_nested_2, zero_shaped

    def test_numpy_sized_dtype_nested_3(self):
        for zero_shaped in (False, True):
            yield self.check_numpy_sized_dtype_nested_3, zero_shaped

    def test_python_collection(self):
        for tp in (list, tuple, set, frozenset, collections.deque):
            yield self.check_python_collection, tp, 'same-dims'
            yield self.check_python_collection, tp, 'diff-dims'

    def test_dict_like(self):
        for tp in self.dict_like:
            yield self.check_dict_like, tp

    def test_dict_like_other_type_key(self):
        # Set the other key types.
        other_tps = ['bytes', 'numpy.bytes_', 'numpy.unicode_',
                     'int', 'float']
        for tp in self.dict_like:
            for other_tp in other_tps:
                yield self.check_dict_like_other_type_key, tp, other_tp

    def test_dict_like_key_null_character(self):
        for tp in self.dict_like:
            yield self.check_dict_like_key_null_character, tp

    def test_dict_like_key_forward_slash(self):
        for tp in self.dict_like:
            yield self.check_dict_like_key_forward_slash, tp

    def test_dict_like_key_back_slash(self):
        for tp in self.dict_like:
            yield self.check_dict_like_key_back_slash, tp

    def test_dict_like_key_leading_periods(self):
        for tp in self.dict_like:
            yield self.check_dict_like_key_leading_periods, tp

    def test_slice(self):
        for _ in range(10):
            data = random_slice()
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def test_range(self):
        for _ in range(10):
            data = random_range()
            out = self.write_readback(data, random_name(),
                                      self.options)
            self.assert_equal(out, data)

    def test_chainmap(self):
        data = random_chainmap()
        out = self.write_readback(data, random_name(), self.options)
        self.assert_equal(out, data)

    def test_fraction(self):
        data = random_fraction()
        out = self.write_readback(data, random_name(), self.options)
        self.assert_equal(out, data)

    def test_dtype(self):
        for dt in {np.dtype(v) for v
                   in itertools.chain(np.sctypeDict,
                                      np.sctypeDict.values())
                   if not isinstance(v, int)
                   and v not in ('V', 'void', 'void0', 'Void0',
                                 np.void)}:
            # Do the test for all endiannesses.
            if dt.byteorder != '|':
                yield self.check_dtype, dt
            else:
                yield self.check_dtype, dt.newbyteorder('<')
                yield self.check_dtype, dt.newbyteorder('>')

    def test_dtype_flexible(self):
        for c in 'SVU':
            for i in range(10):
                if c == 'V' and i == 0:
                    continue
                s = c + str(i)
                self.check_dtype(np.dtype('<' + s))
                self.check_dtype(np.dtype('>' + s))

    def test_dtype_shaped(self):
        base_dtypes = tuple((set(self.dtypes) | {'U2', 'S3'})
                            - {'U', 'S'})
        for i in range(10):
            desc = (random.choice(base_dtypes),
                    random_numpy_shape(random.randint(1, 4), 10))
            self.check_dtype(np.dtype(desc))

    def test_dtype_structured(self):
        base_dtypes = tuple((set(self.dtypes) | {'U2', 'S3'})
                            - {'U', 'S'})
        for i in range(10):
            names = []
            for _ in range(random.randint(1, 5)):
                s = random_str_ascii(random.randint(1, 10))
                while s in names and s[0].isdigit():
                    s = random_str_ascii(random.randint(1, 10))
                names.append(s)
            desc = [(v, random.choice(base_dtypes),
                     random_numpy_shape(random.randint(1, 4), 10))
                    for v in names]
            self.check_dtype(np.dtype(desc))
            self.check_dtype(np.dtype(desc, align=True))

    def test_dtype_structured_with_offsets_titles(self):
        base_dtypes = tuple((set(self.dtypes) | {'U2', 'S3'})
                            - {'U', 'S'})
        for i in range(10):
            names = []
            for _ in range(random.randint(1, 5)):
                s = random_str_ascii(random.randint(1, 10))
                while s in names and s[0].isdigit():
                    s = random_str_ascii(random.randint(1, 10))
                names.append(s)
            titles = []
            for _ in range(len(names)):
                s = random_str_some_unicode(random.randint(1, 10))
                while s in titles:
                    s = random_str_some_unicode(random.randint(1, 10))
                titles.append(s)
            formats = [(random.choice(base_dtypes),
                        random_numpy_shape(random.randint(1, 4), 10))
                       for _ in range(len(names))]
            offsets = [random.randint(0, 100)
                       for _ in range(len(names))]
            desc = {'names': names,
                    'formats': formats,
                    'titles': titles,
                    'offsets': offsets}
            desc_with_itemsize = desc.copy()
            desc_with_itemsize['itemsize'] = \
                np.dtype(desc).itemsize + random.randint(1, 100)
            # Make the dtypes in all combinations of the description and
            # align. Note that if the type isn't valid at all, it is not
            # tested.
            dts = []
            for d in (desc, desc_with_itemsize):
                for align in (False, True):
                    try:
                        dts.append(np.dtype(d, align=align))
                    except:
                        pass
            for dt in dts:
                self.check_dtype(dt)

    def test_datetime_timedelta(self):
        for _ in range(10):
            data = datetime.timedelta(days=random.randint(-20, 20),
                                      seconds=random.randint(-1000,
                                                             1000),
                                      microseconds=random.randint(
                                          -1000**3, 1000**3))
            out = self.write_readback(data, random_name(), self.options)
            self.assert_equal(out, data)

    def test_datetime_timezone(self):
        for _ in range(10):
            data = random_datetime_timezone()
            out = self.write_readback(data, random_name(), self.options)
            self.assert_equal(out, data)

    def test_datetime_date(self):
        for _ in range(10):
            data = datetime.date(year=random.randint(datetime.MINYEAR,
                                                     datetime.MAXYEAR),
                                 month=random.randint(1, 12),
                                 day=random.randint(1, 28))
            out = self.write_readback(data, random_name(), self.options)
            self.assert_equal(out, data)

    def test_datetime_time(self):
        for _ in range(10):
            data = datetime.time(hour=random.randint(0, 23),
                                 minute=random.randint(0, 59),
                                 second=random.randint(0, 59),
                                 microsecond=random.randint(0, 999999),
                                 tzinfo=random_datetime_timezone())
            out = self.write_readback(data, random_name(), self.options)
            self.assert_equal(out, data)

    def test_datetime_datetime(self):
        for _ in range(10):
            data = datetime.datetime(
                year=random.randint(datetime.MINYEAR, datetime.MAXYEAR),
                month=random.randint(1, 12),
                day=random.randint(1, 28),
                hour=random.randint(0, 23),
                minute=random.randint(0, 59),
                second=random.randint(0, 59),
                microsecond=random.randint(0, 999999),
                tzinfo=random_datetime_timezone())
            out = self.write_readback(data, random_name(), self.options)
            self.assert_equal(out, data)


class TestPythonFormat(TestPythonMatlabFormat):
    def __init__(self):
        # The parent does most of the setup. All that has to be changed
        # is turning MATLAB compatibility off and changing the file
        # name.
        TestPythonMatlabFormat.__init__(self)
        self.options = hdf5storage.Options(matlab_compatible=False)


class TestNoneFormat(TestPythonMatlabFormat):
    def __init__(self):
        # The parent does most of the setup. All that has to be changed
        # is turning off the storage of type information as well as
        # MATLAB compatibility.
        TestPythonMatlabFormat.__init__(self)
        self.options = hdf5storage.Options(store_python_metadata=False,
                                           matlab_compatible=False)

        # Add in float16 to the set of types tested.
        self.dtypes.append('float16')

    def assert_equal(self, a, b):
        assert_equal_none_format(a, b, self.options)


class TestMatlabFormat(TestPythonMatlabFormat):
    def __init__(self):
        # The parent does most of the setup. All that has to be changed
        # is turning on the matlab compatibility, and changing the
        # filename.
        TestPythonMatlabFormat.__init__(self)
        self.options = hdf5storage.Options(store_python_metadata=False,
                                           matlab_compatible=True)

    def assert_equal(self, a, b):
        assert_equal_matlab_format(a, b, self.options)
