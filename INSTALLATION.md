## Installation

`awebox` runs on Python 3. It depends heavily on the modeling language CasADi, which is a symbolic framework for algorithmic differentiation. CasADi also provides the interface to the NLP solver IPOPT.  
It is optional but highly recommended to use HSL linear solvers as a plugin with IPOPT.

1.   Get a local copy of the latest `awebox` release:

     ```
     git clone https://github.com/awebox/awebox.git
     ```

2.   Install CasADI version **3.5** for Python 3, following these [installation instructions](https://github.com/casadi/casadi/wiki/InstallationInstructions).

3.   In order to get the HSL solvers and render them visible to CasADi, follow these [instructions](https://github.com/casadi/casadi/wiki/Obtaining-HSL).

##### possible variations to the HSL installation instructions, for linux installation in the /usr/ directory, using anaconda: 
 
- At Instruction  6.  (to avoid Step 2 and the use of LD_LIBRARY_PATH to let IPOPT know where to find libhsl.so, which does not seem to work):
```
./configure --prefix=/usr LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O2 -fopenmp" FCFLAGS="-g -O2 -fopenmp" CFLAGS="-g -O2 -fopenmp"
```
 
- At Instruction 7. (very important, as the metis library is not being linked otherwise) :
```
make LDFLAGS="-lmetis"  
```
 
- At Instruction 8. (because of /usr chosen as install directory)  :
```
sudo make install
```
 

- At Instruction 9. :
```
ln -s usr/lib/libcoinhsl.so (anaconda folder where CasaDI is installed)/libhsl.so
```
The anaconda folder may possibly be at: /home/user_name/anaconda3/lib/python3.7/site-packages/casadi
