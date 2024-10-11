import logging
import pickle
from shutil import rmtree
import unittest
import casadi as ca
import json
import os
from os import path, makedirs, listdir, rename
from typing import Callable
from subprocess import call
import platform

SOURCE_FOLDER = os.getcwd()# path.dirname(path.abspath(__file__))
CACHE_FOLDER = path.join(SOURCE_FOLDER, "cache")

if not path.exists(CACHE_FOLDER):
    makedirs(CACHE_FOLDER)

logger = logging.getLogger(__name__)
if platform == 'Linux':
    _COMPILERS = ["gcc"] # Linux
elif platform == 'Darwin':
    _COMPILERS = ["clang"]  # OSX
elif platform == 'Windows':
    _COMPILERS = ["cl.exe"] # Windows
_COMPILER = None

# Data utils
def write_json(data, file):
    """Write json file."""
    with open(file, "w") as f:
        json.dump(
            data,
            f, indent=4
        )

def read_json(file):
    """Load json file."""
    with open(file, 'r') as f:
        data = json.load(f)
    return data


# Compiler utils
def get_compiler():
    """Get available compiler."""
    global _COMPILER
    if _COMPILER is None:
        for compiler in _COMPILERS:
            try:
                call([compiler, "--version"])
                _COMPILER = compiler
                break
            except Exception:
                pass

    return _COMPILER

def compile(input_file, output_file, options=None):
    """Compile a c file to an so file."""
    compiler = get_compiler()

    _CXX_FLAGS = ["-fPIC", "-shared", "-fno-omit-frame-pointer", "-O1"] # add "-v" for verbose
    call([compiler] + _CXX_FLAGS + ["-o", output_file, input_file])

class CachedFunction:
    """A cached function."""

    def __init__(self, name, func, filename=None, do_compile=None):
        """Load or create a cached function."""
        if do_compile is None:
            do_compile = True
        self.name = name + "_" + func.name()
        if filename is None:
            self.filename = path.join(CACHE_FOLDER, self.name)
        else:
            self.filename = filename

        if self._exist_so():
            logger.debug(f"Loading function from so-file {self.name}")
            self._load_so()
        else:
            logger.debug(f"Creating function {self.name}")
            self.f = func
            if do_compile:
                logger.debug(f"Compiling function {self.name}")
                self._save_so()
                self._load_so()

    def _create(self, func):
        """Create a function."""
        self.f = func

    def _exist_so(self):
        """Check if the file exist."""
        return path.exists(self.filename + ".json")

    def _save_so(self):
        """Save as so file."""
        data = {
            "lib_file": self.filename + ".so",
            "c_file": self.filename + ".c",
            "func_name": self.f.name()
        }

        tmp_file = data["func_name"] + ".c"
        cg = ca.CodeGenerator(tmp_file)
        cg.add(self.f)
        cg.add(self.f.jacobian())
        cg.add(self.f.jacobian().jacobian())
        cg.generate()
        rename(tmp_file, data["c_file"])
        compile(data["c_file"], data["lib_file"])
        write_json(data, self.filename + ".json")

    def _load_so(self):
        """Load an SO file."""
        data = read_json(self.filename + ".json")
        self.f = ca.external(data["func_name"], data["lib_file"])

    def __call__(self, *args, **kwargs):
        """Call the function."""
        return self.f(*args, **kwargs)
    
    def map(self, *args, **kwargs):
        """Create map of the function."""
        return self.f.map(*args, **kwargs)

# Below testing for a simple example
def dummy_casadi_function():
    """Create a simple function."""
    x = ca.SX.sym("x", 2)
    f = x[0] ** 2 + x[1]
    return ca.Function("f", [x], [f])

if __name__ == "__main__":
    unittest.main()