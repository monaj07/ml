#
import unittest
import numpy as np
import sys
sys.path.append("..")
from ml.classification import sig


class TestLogistic(unittest.TestCase):
    def test_sig(self):
        input = 1
        expected_out = 1 / (1 + np.exp(-input))
        output = sig(input)
        self.assertEqual(output, expected_out)
