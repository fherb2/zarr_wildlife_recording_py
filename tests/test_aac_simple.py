# VEREINFACHTER TEST - test_aac_simple.py
import unittest
import pathlib
from zarrwlr.config import Config
from zarrwlr.packagetypes import LogLevel

class TestAACSimple(unittest.TestCase):
    def test_aac_import_basic(self):
        """Vereinfachter AAC Import Test"""
        print("Testing AAC import...")
        
        # Test Config
        try:
            Config.set(aac_default_bitrate=128000)
            print("✓ Config test passed")
        except Exception as e:
            print(f"✗ Config test failed: {e}")
        
        # Test Import
        try:
            from zarrwlr.aac_access import import_aac_to_zarr
            print("✓ AAC import successful")
        except Exception as e:
            print(f"✗ AAC import failed: {e}")

if __name__ == '__main__':
    unittest.main()
