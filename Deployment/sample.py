import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import test_sample

def test_hello(capfd):
    sample.hello()
    out, _ = capfd.readouterr()
    assert "Hello" in out
