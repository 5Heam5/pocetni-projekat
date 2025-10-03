import unittest
from math_utils import add_numbers

class TestMathUtils(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add_numbers(2, 3), 5)

if __name__ == '__main__':
    unittest.main()
