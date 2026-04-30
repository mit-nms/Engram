import os
import sys

# Add the vidur package directory to Python path
vidur_dir = os.path.dirname(os.path.abspath(__file__))
vidur_dir = os.path.join(vidur_dir, "vidur_repo")
sys.path.insert(0, vidur_dir)

# Add the vidur package directory
vidur_pkg_dir = os.path.join(vidur_dir, "vidur")
sys.path.insert(0, vidur_pkg_dir)
