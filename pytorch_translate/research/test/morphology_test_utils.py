#!/usr/bin/env python3

import tempfile
from os import path


def get_two_same_tmp_files():
    src_txt_content = [
        "123 124 234 345",
        "112 122 123 345",
        "123456789",
        "123456 456789",
    ]
    dst_txt_content = [
        "123 124 234 345",
        "112 122 123 345",
        "123456789",
        "123456 456789",
    ]
    content1, content2 = "\n".join(src_txt_content), "\n".join(dst_txt_content)
    tmp_dir = tempfile.mkdtemp()
    file1, file2 = path.join(tmp_dir, "test1.txt"), path.join(tmp_dir, "test2.txt")
    with open(file1, "w") as f1:
        f1.write(content1)
    with open(file2, "w") as f2:
        f2.write(content2)

    return tmp_dir, file1, file2


def get_two_different_tmp_files():
    src_txt_content = [
        "123 124 234 345",
        "112 122 123 345",
        "123456789",
        "123456 456789",
    ]
    dst_txt_content = [
        "abc abd bcd cde",
        "aab abb abc cde",
        "abcdefghi",
        "abcdef defghi",
    ]
    content1, content2 = "\n".join(src_txt_content), "\n".join(dst_txt_content)
    tmp_dir = tempfile.mkdtemp()
    file1, file2 = path.join(tmp_dir, "test1.txt"), path.join(tmp_dir, "test2.txt")
    with open(file1, "w") as f1:
        f1.write(content1)
    with open(file2, "w") as f2:
        f2.write(content2)

    return tmp_dir, file1, file2
