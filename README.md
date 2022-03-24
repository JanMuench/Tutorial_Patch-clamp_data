# Tutorial for Patch-clamp data analysis
## General Info
This tutorial contains the example code for the analysis of patch-clamp measurements which is part of the publication “Bayesian inference of kinetic schemes for ion channels by Kalman filtering”. The work was done in Stan https://mc-stan.org/ with the PyStan https://pystan.readthedocs.io/en/latest/ interface version 2.19.1.2 which is currently outdated. We plan to update the tutorial to PyStan 3 in the near future. The code is parallelized to use multiple CPUs of a node on a compute cluster. Each individual sampling chain is trivially calculated in parallel using the standard functionalities provided by the Stan language.

The package contains a file containing the Stan code [“KF.txt”](KF.txt) as well as the Python script [“compile_CCCCO_normal_split.py”](compile_CCCCO_normal_split.py) to compile the code.  Finally, the Python script [“sample_PC_data.py”](sample_PC_data.py) acts as the interface between the data and the sampler. To adapt the code to your data, basic Stan programming skills are required. Python knowledge and the Python scripts are not obligatory, because Stan can interact with many high level data analysis programming languages (R, Python, shell, MATLAB, Julia, Stata).  

The topology of the kinetic scheme is uniquely defined by a rate matrix. Our example code demonstrates the analysis with a two-ligand-gated 4-state model of patch-clamp data. The rate matrix is defined by the [lines 543-563](KF.txt#L535) in the file “KF.txt”. The mean observation matrix is defined in [line 806](KF.txt#L810) with the vector variable `conduc_state`. 

## Step by step:

<details>
<summary><b> How to start the posterior sampling of the example PC data </b></summary>

1. One needs to install Stan and PyStan.

2. One executes “compile_CCCCO_normal_split.py” by entering
```console
python3 compile_CCCCO_normal_split.py
```
into the command line.
That compiles the Stan code [“KF.txt”](KF.txt) into an executable program `KF_CCCO.pic`.

3. Entering
```console
python3 sample_PC_data.py 8000
```
executes a Python program which acts as an interface between the data from [“data/current8000.npy”](data/current8000.npy) and 	    
sampling algorithm `KF_CCCO.pic`. In the folder, data are 4 numpy arrays. The numpy 
array [“data/current8000.npy”](data/current8000.npy) has the data of 10 different ligand concentrations with two
ligand jumps from zero to the concentration and back to zero. The numpy array [“Time.npy”](data/Time.npy)
is the time axis of all traces in the current array. The ligand concentrations are saved 
in [“ligand_conc.txt”](data/ligand_conc.txt) and [“ligand_conc_decay.txt”](data/ligand_conc_decay.txt). 
Each row of the ligand matrix defines an array whose entries are element-wise multiplied with the rates in the function 
`multiply_ligandconc_CCCO`. Ligand-independent rates are multiplied by one and the 
ligand-depended rates are multiplied with a ligand concentration.
The time points of the concentration jumps are defined in the script  [“sample_PC_data.py”](sample_PC_data.py) 
in the functions [data_slices_beg_new](sample_PC_data.py#L51) and [data_slices_decay_new](sample_PC_data.py#L115)
Additionally, each time trace is cut such that activation or deactivation is treated as an individual time trace on an 
individual CPU.
We assumed that we only needed 5 patches. So two ligand concentrations 
were measured from one patch. For optimal caluclation efficiency, 10 time traces 
require 20 CPUs (activation and decay) or 40 CPUs to apply cross-validaton 4 times to 
4 independent sample chains.


</details>



<details>
<summary><b>The output of samples as we used in the publication</b></summary>
- The csv file `rate_matrix_params` contains the samples of the posterior of the rate 
	  matrix. Simply analysing them means that we marginalized out all the other parameters. Note
	  that the dwell times are on a scaled log scale thus one has to multiply them by a 
	  scaling factor for the actual log scale. 
- The single-channel current samples are saved in an numpy array `i_single.npy`.
- The samples of the variance parameter are saved in the numpy file `measurement_sigma.npy`.
- The samples of the open-channel variance parameter are saved in the numpy file `open_variance.npy`.
- The samples of the “Ion channels per time trace parameter” are saved in the numpy file `N_traces.npy`.

</details>

<details>
<summary><b>How to adapt the kinetic scheme to new data</b></summary>

5. To adapt the kinetic scheme, one needs to change two matrices inside [“KF.txt”](KF.txt): the rate marix and observation
matrix which defines which states are conducting and the functions related to the kinetic scheme. After all
changes to the Stan program, “KF.txt” needs to be recompiled.

	1. The function
	```Stan
	matrix create_rate_matrix(real[] theta_array,
			          real[] ratios,
				  int N_free_para,
				  vector ligand_conc,
				  int M_states,
				  real numeric_precision)
	{

		matrix[M_states, M_states] rates;
		vector[N_free_para] theta_vec = multiply_ligandconc_CCCO_log_uniform(theta_array,
							ratios,
							N_free_para,
							ligand_conc);

		rates = assign_param_to_rate_matrix_CCCO(theta_vec, M_states);

		rates  = assign_diagonal_elements(rates, M_states, numeric_precision);

		return rates;
	 }
	```
	defines the rate matrix:
	First, the function `multiply_ligandconc_CCCO` needs to be adapted. That function takes the parameters 		
	from the parameters block and computes the rates of the rate matrix:

	```Stan
	vector multiply_ligandconc_CCCO_log_uniform(real[] theta_array,
		                                    real[] equili,
		                                    int N_free_para,
		                                    vector ligand_conc)
	{

		vector[N_free_para] theta;
		//print("ratio: ", theta_array[6]);
		theta[2] = theta_array[1];

		theta[4] = theta_array[2] ;
		theta[1] = theta[4] / (1 - equili[1]) * equili[1];

		theta[3] = theta_array[3] * equili[2];
		theta[6] = theta_array[3] * (1 - equili[2]);
		theta[5] = pow(10, (4.7 * equili[3] - 1));

		return theta .* ligand_conc;
	}
	```


	There some rates in our example whose value scale linearly with the ligand concentration.
	Ad the end of the function (line 88) the rates are mutliplied elementwise with the respective
	ligand concentration or simply with one if they are not ligand-concentration-dependent. The return 
	variables are then passed to the `assign_param_to_rate_matrix_CCCO` function. Note that this 
	example code has four dwell times as transition parameters and two probabilities from which the six 
	rates are constructed. We recommend to use a log-uniform prior for the dwell time and a beta distribution or
	rather a Dirichlet distribution for the probabilities whose transitions are taken.
	The function gets the information which rate is ligand-concentration-dependent from a array 
	which consist of entries which equal ones and entries which equal the ligand concentration.
	Note, that for each ligand concentration here exist one array which gets distributed to the CPU on the upper
	level of the Stan programm. Thus on this level every function is programme just if there was only one ligand
	concentration. The arrays of ligand concentrations are defined in
	the ["data" folder](data/) in the files [ligand_conc.txt](data/ligand_conc.txt) for the activation and 
	[ligand_conc_decay.txt](data/ligand_conc_decay.txt) for the deactivation

	The rate matrix is defined in the next following function `assign_param_to_rate_matrix_CCCO` in
	"create_rate_matrix" (line 61).
	
	As an example the function:
	```Stan
	matrix assign_param_to_rate_matrix_CCCO(vector theta, int M_states)
	{
		matrix[M_states, M_states] rates_mat;
		rates_mat    = [[      0 , theta[1],        0,         0],
				[theta[2],        0, theta[3],         0],
				[       0, theta[4],        0,  theta[5]],
				[       0,        0,  theta[6],        0]];



		return rates_mat;
	}
	```
	gets the vector variable `theta` with the rates and an int variable `M_states`
	with the number of Markov states. "M_states is" It defines the topology of the kinetic scheme by
	the independent non-zero coefficients. Thus we defined here a 4x4 rate matrix with 
	6 chemical reaction channels which describe the kinetic scheme of the ion channel.
	Each ion channel has 2 states it is directly connected with by one transition 
	(monomolocular chemical  reaction). Only the first and the fourth state have only 
	one ajacent state. We chose the notation where the matrix acts onto the a column vector to its right 
	which means the each coloumn of the rate matrix needs to be sum to zero. This happens 
	in the following function `assign_diagonal_elements(rates, M_states, numeric_precision);`.

	To change the topology of the kinetic scheme from a 4-state to a 5-state kinetic scheme 
	with a loop structure, we change the function (and rename it):
	```Stan
	matrix assign_param_to_rate_matrix_CCO_CO(vector theta, int M_states)
	{
		matrix[M_states, M_states] rates_mat;
		rates_mat    = [[      0 , theta[1],        0,         0,             0],
				[theta[2],        0, theta[3],         0,      theta[7]],
				[       0, theta[4],        0,  theta[5],             0],
				[       0,        0,  theta[6],        0,      theta[9]],
				[       0, theta[8],        0, theta[10],             0]];



		return rates_mat;
	}
	```

	Remember that each i-th row shows the transitions out of i-th state. Thus you can read 
	from this matrix:
	
	* The first state transitions into the second. 	
	* The second transitions into the first, the third, and the fifth.
	* The third transitions into the second and fourth.
	* The fourth into the third and the fifth
	* The fifth into the second and fourth.
		
	Now, obvisously that we changed the function name which defines the kinetic scheme we have to change 
	the name also in the place where the function is called.
	So instead of `assign_param_to_rate_matrix_CCCO` here in line 61 we have to change it to
	`assign_param_to_rate_matrix_CCO_CO` in the KF.txt file. The KF.txt file gets the number of Markov states
	as an input from the python script which starts the sampling.

	As mentioned above the function `assign_param_to_rate_matrix_CCCO` assigns rates to the off-diagonal elements. Note
	that a closed first-order Markov system requires that each diagonal element is the negative sum of its column. 
	That property is enforced in function `assign_diagonal_elements`. Note that this is redundant as we start 
	in the parameters block with the dwell times as parameters. But we could have chosen a different 
	parametrization to begin with. In a current project we investiage this parametrisation but there are a couple of other
	options.


</details>

<details>
<summary><b>Changes of the observation model</b></summary>

	2. The row vector `conduc_state` needs to  be changed to the desired signal model. It represents the 
	matrix H from the article which generates the mean signal for a given ensemble state but also adds covariance 
	to signal due the fact that the true system state is unkown. In the function `calcLikelihood_for_each_trace` in (line
	[KF.txt](KF.txt#L794) we defined the linear observation matrix
	as a row vector whose:
	```Stan
	row_vector[M_states] conduc_state = [0,0,0, i_single_channel];
	```
	The fourth state is in this case the conducting state. Every other of the three states has a conductance of zero.
	If more than two conducting classes (non-conducting and conducting) are modeled, additional single-channel current
	parameters need to be defined in the parameters block.

	3. If there are multiple open-channel noise standard deviations states
	the function `calc_sigma_and_mean` must be adapted.

</details>


	



## Some other comments

Although we recommend to have the dwell times (diagonal elements of the rate matrix) as parameters, we recalculate them which is reminiscent of former parameterizations.  

Note, that PyStan 3 is not backwards compatible. Thus, using PyStan 3 requires some minor changes to the compilation, sampling, and saving of the samples. To visualize and analyze a posterior/draw from the posterior, we recommend the package Corner.py. To implement posterior post-processing and diagnosis in a Bayesian workflow, we highly recommend  using the Arviz package.

Note, that in the first KF analysis round we do not report the derived quantities such as mean signal and covariance for a given time.  We discussed in the Appendix of the acticle that this would expand the total runtime of the program by roughly two orders of magnitude.  To show the posterior of the mean signal and the posterior of the variance, we suggest to use a subset of the posterior samples and feed it to the KF to do the filtering. This requires minimal changes to the KF code.

The data used for the posterior is `data_start.npy` and `data_dec.npy`. The suffix “hold” means that this would be the data used as a hold-out data set if one did cross-validation.

			
