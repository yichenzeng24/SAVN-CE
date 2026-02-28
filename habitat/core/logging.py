#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Modifications Copyright (c) 2026 Yichen Zeng, Wuhan University, Email: zengyichen@whu.edu.cn
# Description: Adapted for semantic audio-visual navigation in continuous environment (SAVN-CE).

import logging
import os


# class HabitatLogger(logging.Logger):
#     def __init__(
#         self,
#         name,
#         level,
#         filename=None,
#         filemode="a",
#         stream=None,
#         format_str=None,
#         dateformat=None,
#         style="%",
#     ):
#         super().__init__(name, level)
#         if filename is not None:
#             handler = logging.FileHandler(filename, filemode)  # type:ignore
#         else:
#             handler = logging.StreamHandler(stream)  # type:ignore
#         self._formatter = logging.Formatter(format_str, dateformat, style)
#         handler.setFormatter(self._formatter)
#         super().addHandler(handler)

#     def add_filehandler(self, log_filename):
#         filehandler = logging.FileHandler(log_filename)
#         filehandler.setFormatter(self._formatter)
#         self.addHandler(filehandler)

class HabitatLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)  # type:ignore
        else:
            handler = logging.StreamHandler(stream)  # type:ignore
        self._formatter = logging.Formatter(format_str, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)
        self._filehandlers = {} 

    def add_filehandler(self, log_filename, rank=[0]):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank not in rank or log_filename in self._filehandlers:
            return
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)
        self._filehandlers[log_filename] = filehandler

logger = HabitatLogger(
    name="habitat", level=logging.INFO, format_str="%(asctime)-15s %(message)s"
)
