# Tutorial_Patch-clamp_data
This tutorial is an example code for patch-clamp measurements which is part of the publication “Bayesian inference of kinetic schemes for ion channels by Kalman filtering”. The work was done in Stan https://mc-stan.org/ with the PyStan https://pystan.readthedocs.io/en/latest/ interface version 2.19.1.2 which is currently outdated. We plan to update the tutorial folder to PyStan 3 in the near future. The code is parallelized for the multiple CPUs of a node on a computation cluster. Each individual sampling chain is trivially calculated in parallel provided by the Stan language.

The package contains a file containing the STAN code “KF.txt” as well as the python script  “compile_CCCCO_normal_split.py” to compile the code.  Finally, a Python script “sample_PC_data.py” which acts as the interface between the data and the sampler. To adapt the code to your data, basic Stan programming skills are required. The Python knowledge and the Python scripts are not obligatory, because Stan can interact many high level data analysis programming languages (R, Python, shell, MATLAB, Julia, Stata) .  

The topology of the kinetic scheme is uniquely defined by a rate matrix. Our example code demonstrates the analysis with a two ligand-gated 4 state model of patch-clamp data. The rate matrix is defined by the lines 543-563 in the file “KF.txt”. The mean observation is defined in line 806 with the vector variable “conduc_state”. 

Step by step:

	1. One needs to install Stan and PyStan.
	2. One executes “compile_CCCCO_normal_split.py” by prompting
		 “python3  compile_CCCCO_normal_split.py” into the command line.
		 That compiles the Stan code “KF.txt” into an executable program “KF_CCCO.pic”.
	3. Prompting “python3 sample_PC_data.py 8000” executes a Python program which acts as an interface between the data from “data/current8000.npy” and 			        sampling algorithm “KF_CCCO.pic”. In the folder, data are 4 numpy arrays. The numpy array “current8000.npy” has the data of 10 different 	              	   ligand concentrations with two ligand jumps from zero to the concentration and back to zero. The numpy array  “Time.npy” is the time axis of all 			        traces in the  current array. The ligand concentrations are saved in “ligand_conc.txt” and “ligand_conc_decay.txt”. Each row of the ligand matrix defines      an array whose entries are element-wise multiplied to the rates in the function 
