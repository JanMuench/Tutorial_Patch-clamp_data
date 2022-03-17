# Tutorial_Patch-clamp_data
This tutorial is an example code for patch-clamp measurements which is part of the publication “Bayesian inference of kinetic schemes for ion channels by Kalman filtering”. The work was done in Stan https://mc-stan.org/ with the PyStan https://pystan.readthedocs.io/en/latest/ interface version 2.19.1.2 which is currently outdated. We plan to update the tutorial folder to PyStan 3 in the near future. The code is parallelized for the multiple CPUs of a node on a computation cluster. Each individual sampling chain is trivially calculated in parallel provided by the Stan language.

The package contains a file containing the STAN code “KF.txt” as well as the python script  “compile_CCCCO_normal_split.py” to compile the code.  Finally, a Python script “sample_PC_data.py” which acts as the interface between the data and the sampler. To adapt the code to your data, basic Stan programming skills are required. The Python knowledge and the Python scripts are not obligatory, because Stan can interact many high level data analysis programming languages (R, Python, shell, MATLAB, Julia, Stata) .  

The topology of the kinetic scheme is uniquely defined by a rate matrix. Our example code demonstrates the analysis with a two ligand-gated 4 state model of patch-clamp data. The rate matrix is defined by the lines 543-563 in the file “KF.txt”. The mean observation matrix is defined in line 806 with the vector variable “conduc_state”. 

Step by step:

	1. One needs to install Stan and PyStan.
	
	2. One executes “compile_CCCCO_normal_split.py” by prompting
	“python3  compile_CCCCO_normal_split.py” into the command line.
	That compiles the Stan code “KF.txt” into an executable program “KF_CCCO.pic”.
		 
	3. Prompting “python3 sample_PC_data.py 8000” executes a Python program which acts 
	as an interface between the data from “data/current8000.npy” and 	    
	sampling algorithm “KF_CCCO.pic”. In the folder, data are 4 numpy arrays. The numpy 
	array “current8000.npy” has the data of 10 different ligand concentrations with two
	ligand jumps from zero to the concentration and back to zero. The numpy array  “Time.npy”
	is the time axis of all traces in the  current array. The ligand concentrations are saved 
	in “ligand_conc.txt” and “ligand_conc_decay.txt”. Each row of the ligand matrix defines an 
	array whose entries are element-wise multiplied to the rates in the function 
	“multiply_ligandconc_CCCO”. Ligand-independent rates are multiplied by one and the ligand
	depended rates are multiplied with a ligand concentration. Within the script  
	“sample_PC_data.py” in the functions “data_slices_beg_new” and 	“data_slices_decay_new” 
	the time points of the concentration jumps are defined. Additionally, each time trace 
	is cutted that activation or deactivation is treated as an individual time trace on an 
	individual CPU. We assumed that we only needed 5 patches. So two ligand concentrations 
	were measured from one patch. For optimal caluclation efficiency, 10 time traces 
	require 20 CPUs (activation and decay). 40 CPU to apply cross validaton times 4 for 
	4 independent sample chains.
	
	4. The output of samples as we used them in the publication.
	4.1 The csv file “rate_matrix_params” saves the samples of the posterior of the rate 
	matrix. Simply analysing them means that we marginalized all other parameters out. Note
	that the dwell times are on a scaled log space 	thus one has to multiply them by a 
	scaling factor for the actual log space. 
	4.2 The single-channel current samples are saved in an numpy array “i_single.npy”.
	4.3 The samples of the variance parameter are saved in the numpy array 	file “measurement_sigma.npy”.
	4.4 The samples of the open-channel variance parameter are saved in the numpy array file “open_variance.npy”.
	4.5 The samples of the “Ion channels per time trace parameter” are saved in the numpy array file “N_traces.npy”.

	5. To adapt the kinetic scheme one needs to change a few things within KF.txt  which are 
	the observation model matrix H and the functions related to the kinetic scheme. Then 
	“KF.txt” needs to be recompiled. The rate matrix is defined in the function "create_rate_matrix" (line 546)
		As an example the function:
			matrix assign_param_to_rate_matrix_CCCO(vector theta, int M_states)
    			{
        			matrix[M_states, M_states] rates_mat;
        			rates_mat    = [[      0 , theta[1],        0,         0],
                        			[theta[2],        0, theta[3],         0],
                        			[       0, theta[4],        0,  theta[5]],
                        			[       0,        0,  theta[6],        0]];



        			return rates_mat;
    			}
	
		gets the vector variabel "theta" with the rates and an int variabel "M_states"
		with the Number of Markov states. It defines the topology of the kinetic scheme by
		the independent non-zero coefficients. Thus we defined here a 4x4 rate matrix with 
		6 chemical reaction channels which describe the kinetic scheme of the ion channel.
		Each ion channel has 2 states it is directly connected with by one transition (graph)
		Only the first and the fourth state one only one ajacent state.
		We chose the notation where the matrix acts onto the a column vector to its right 
		which means the each coloumn of the rate matrix needs to be sum to zero. This happens 
		in the following function "assign_diagonal_elements(rates,M_states, numeric_precision);".
		
		To change the topologie of the knietic scheme from a 4 state to 5 state knietic scheme 
		with a loop structure you could define a function such as this
		
			matrix assign_param_to_rate_matrix_CCO_CO(vector theta,
                                   int M_states)
    			{
        			matrix[M_states, M_states] rates_mat;
        			rates_mat    = [[      0 , theta[1],        0,         0,             0],
                        			[theta[2],        0, theta[3],         0,      theta[7]],
                        			[       0, theta[4],        0,  theta[5],             0],
                        			[       0,        0,  theta[6],        0,      theta[9]],
                        			[       0, theta[8],        0, theta[10],             0]];



        			return rates_mat;
    			}
		
		Remember that each i-th row shows you the transitions out of i-th state. Thus you can read 
		from this matrix: The first state transitions into the second. 	
				  The second transitions into the first, the third and the fifth
				  The third trsnasitions into the second and forth.
				  The forth into the third and the fifth
				  The fifth into the second and fourth.
		
		Now obvisously that we changed the fuction name which defines the kinetic scheme 
		we have to change the name also in the place where the function is called
		
			matrix create_rate_matrix(real[] theta_array,
                           			  real[] ratios,
                           			  int N_free_para,
                           			  vector ligand_conc,
                        			  int M_states,
                        			  real numeric_precision)
		      	{

        			matrix[M_states,M_states] rates;
        			vector[N_free_para] theta_vec = multiply_ligandconc_CCCO_log_uniform(theta_array,
                           				ratios,
                           				N_free_para,
                           				ligand_conc);

        			rates = assign_param_to_rate_matrix_CCCO(theta_vec, M_states);

        			rates  = assign_diagonal_elements(rates,M_states,
                                    numeric_precision);

        			return rates;
    			}
    
    So instead of "assign_param_to_rate_matrix_CCCO" here in line 112 we have to change it to 				     "assign_param_to_rate_matrix_CCO_CO" in KF.txt file. The KF.txt file gets the number of Markov states
    as an input from the python script which starts the the sampling.
    
    There some rates in our example whose value scales linearly with the ligand concentration we calculate these aspects     in the very first function call "multiply_ligandconc_CCCO_log_uniform" in the function "create_rate_matrix"
    The function gets the parameters which define the rate matrix in the following and a array 
    which consist of entries which equal ones and entries which equal the ligand concentration.
    For each ligand concentration one array. The arrays of igand concentrations are define in
    the folder "data" in the files ligand_conc.txt for the acivation and ligand_conc_decay.txt for the deactivation
    
		
	
     5.1. The row vector “conduc_state” needs to  be changed to the desired signal model. It 
	represents the matrix H of the 	article which generates the mean signal for a given 
	ensemble state but also adds covariance to signal due the fact that the true system state is unkown.
	In the function "calcLikelihood_for_each_trace" in line 812 we defined the linear observation matrix
	as a row vector whose:
		row_vector[M_states]      conduc_state = [0,0,0, i_single_channel];
	The fourth state is in this case the conducting state.
	
	If more than  two conducting classes (non-conducting and conducting) are
	modeled, additional single-channel current parameters need to be defined in the parameters block.

	5.2 The function “multiply_ligandconc_CCCO” needs to be adapted. That function takes the parameters from
	the parameters block and computes the rates of the rate matrix. They are then passed to the 
	“assign_param_to_rate_matrix_CCCO” function. Note that this example code has four dwell times as parameters and two 
	ratios from them the six rates are constructed. We recommend to use a log uniform prior for the 
	dwell time and a beta distribution or rather a Dirichlet distribution for the 	probabilities which transition is taken.
	5.3 The function “assign_param_to_rate_matrix_CCCO” assigns rates to 	the off diagonal elements. Note that 
	a closed first order Markov system requires that each diagonal element is the negative sum of its column. 
	That property is enforced in function “assign_diagonal_elements”. Note that this is redundant as we start 
	in the 	parameters block with the dwell times as parameters. But we could have chosen a different 
	parametrization to begin with. We argue in the paper to use this parametrisation in order to use a 
	Jeffreys prior but there a couple of other options.
	5.4 The mean observation needs to be changed in line 806
	5.5 If the amount of open-channel states with differing open-channel noise variances for each state needs to be calculated,
	the function “calc_sigma_and_mean” must be adapted


Although we recommend to have the dwell times (diagonal elements of the rate matrix) as parameters, we recalculate them which is reminiscent of former parameterizations.  

Note, that PyStan 3 is not downward compatible. Thus, using PyStan 3 requires some minor changes to the compiling sampling and saving of the samples. To visualize and analyze a posterior/draw from the posterior, we recommend the package Corner.py. To implement a posterior post processing and diagnosis in a Bayesian workflow, we highly recommend  using the Arviz package.

Note, that in the first KF analysis round we do not report the derived quantities such as mean signal and covariance for a given time.  We discussed in the Appendix of the paper that this would expand the total runtime of the program by roughly two orders of magnitude.  To show the posterior of the mean signal and the posterior of the variance, we suggest to use a subset of the posterior samples and feed it to the KF to do the filtering. It requires minimal changes to the KF code.

The data used for the posterior is “data_start.npy” and “data_dec.npy”. The suffix “hold” means that this would be the data used as a hold-out data set if one would do cross validation.

			
