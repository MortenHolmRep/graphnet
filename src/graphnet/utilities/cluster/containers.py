"""test."""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from builtins import range
from builtins import dict
from future import standard_library

from builtins import object

standard_library.install_aliases()


def trim_duplicates(a_list):  # type: ignore
    """test."""
    output_list = list()
    unique_elements = set()
    for val in a_list:
        if (
            val not in unique_elements
        ):  # e.g. is this value has not already been seen
            output_list.append(val)
            unique_elements.add(val)
    return output_list


#
# Smart containers
#
