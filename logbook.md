# 30/06/2025

 - Meeting with the team.
 - reading :
   1. J. de Schutter's thesis, chap. 2-3
   1. Airborne Wind Energy chap. 1 (Loyd's law)
   1. material for Numerical Optimal Control 2024
   1. Lecture Notes on Numerical Optimization (2017)

 - Project :
   1. Use AWEbox to simulate a one-eight pumping cycle with 2 reel-in and 2-reel-out
   1. Arm model in AWEbox
 
 - Installing AWEbox:
   1. `brew install python@3.9`
   1. `brew install python-tk@3.9`
   1. `cd awebox`
   1. `python3.9 -m venv venv`
   1. `venv/bin/pip3 install .`

 - Installing HSL
   1. `brew install gcc`
   1. `brew install gfortran`
   1. `brew install metis`
   1. Download `coinhsl-2022.11.09.zip` from https://licences.stfc.ac.uk/account/downloads and extract
   1. `git clone https://github.com/coin-or-tools/ThirdParty-HSL.git`
   1. `cd ThirdParty-HSL`
   1. Unpack `coinhsl-2022.11.09.zip` to `ThirdParty-HSL/coinhsl`
   1. necessary ? `brew install llvm libomp`
   1. `export DIR=/usr/local`
   1. add the following to `.zprofile`: ```
export CC=/opt/homebrew/bin/gcc-15
export CXX=/opt/homebrew/bin/g++-15
alias gcc=/opt/homebrew/bin/gcc-15
alias cc=/opt/homebrew/bin/gcc-15```
   1. Debugging with Gemini: `
./configure --prefix=$DIR \
  --with-metis-cflags="-I$(brew --prefix metis)/include" \
  --with-metis-lflags="-L$(brew --prefix metis)/lib -lmetis" \
  CXXFLAGS="-g -O2 -fopenmp" \
  CFLAGS="-g -O2 -fopenmp" \
  FCFLAGS="-g -O2 -fopenmp" \
  LDFLAGS="-fopenmp"
 `
  1. `make`, `sudo make install`
  1. `sudo ln -s $DIR/lib/libcoinhsl.dylib $DIR/lib/libhsl.dylib`
  1. restore bash profile
  1. Add `export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/gcc/15.1.0/lib/gcc/15/:$DYLD_LIBRARY_PATH` to `.zprofile` (path found by `locate libgfortran.5.dylib`)

path to libgfortran.5.dylib
/opt/homebrew/Cellar/gcc/15.1.0/lib/gcc/15/libgfortran.5.dylib
export DYLD_LIBRARY_PATH="/usr/local/lib:/opt/homebrew/lib:$DYLD_LIBRARY_PATH"


A essayer :
 - build and install from source https://github.com/casadi/casadi/issues/2829, être sûr que les binaires sont OK
 - start with a fresh gemini session


   1. `.venv/bin/python examples/ampyx_ap2_trajectory.py` -> Error ipopt not found
