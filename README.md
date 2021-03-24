# Ising_Model_2D

Current project undertakes asessment of two-dimensional Ising Model. The model represents a many-body system and can be solved with various approaches. The evolution of the model is dictated by the flip of its spins. There are various methods to simulate the problem. 

In this project I examine the Metropolis Algorithm, which belongs to a family of Monte Carlo simuation. 

To eliminate the slowing down around the critical temperature and to speed up the general execution time - a Swendsen-Wang cluster algorithm is introduced.


## Structure

Use _run.py_ to initiate different simulations. Edit simulation-specific settings in the _run _ __simulation__.py_ scripts. Metropolis algorithm is set up with cython, therefore one needs to make sure that all appropriate packages are installed before proceeding. A _setup.py_ file is used to compile a file of the type '.pyx'. Instructions on how to perform compilation are inside the script.
