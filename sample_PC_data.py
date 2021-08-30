#!/afs/rz.uni-jena.de/home/n/nu65jem/myenv/bin/python
import numpy as np
import time
import datetime
import sys
import pickle
import os
#from get import get_mean
#from fit_and_load_stan_model import create_model_and_fit
from save_sampling import save_data_new as save



def load(filename):
    """Reload compiled models for reuse."""
    print("Trying to load pickle in:")
    print(os.getcwd())
    return pickle.load(open(filename,'rb'))

def create_model_and_fit(DATA, name, sampling_iter, warmingUp, chains):
    print("get model and fit:"+os.getcwd())
    try:
        model = load(name)
    except:
        model = load("RE_approach.pic")
    print("sampling_iter", sampling_iter)
    print("sampling in: " + os.getcwd())
    print("warmup"+str(warmingUp))
    print("chains"+str(chains))

    samples_posterior = model.sampling(DATA,
                         n_jobs = -1,
                         chains=chains,
                         thin=1,
                         warmup=warmingUp,#4000,
                         iter=int(sampling_iter),
                         verbose=True,
                         refresh = 400,
                         test_grad = None)

    print("finished sampling")
    try:
        samples_posterior.summary()
    except:
        print("could not create fit summary")

    return samples_posterior, model



def data_slices_beg_new(data, Time, skip):
    data = data.swapaxes(0,1)
    y_1 = data[2700:3100:int(400/skip),0]
    y_2 = data[2600:3000:int(400/skip),1]
    y_3 = data[2580:2980:int(400/skip),2]
    y_4 = data[2540:2940:int(400/skip),3]
    y_5 = data[2520:2920:int(400/skip),4]
    y_6 = data[2516:2916:int(400/skip),5]
    y_7 = data[2510:2910:int(400/skip),6]
    y_8 = data[2510:2910:int(400/skip),7]
    y_9 = data[2504:2904:int(400/skip),8]
    y_10 = data[2503:2903:int(400/skip),9]


    after_jump = np.array([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])

    print(after_jump)

    print(data.shape)

    y_1 = data[2410,0]
    y_2 = data[2410,1]
    y_3 = data[2410,2]
    y_4 = data[2410,3]
    y_5 = data[2410,4]
    y_6 = data[2410,5]
    y_7 = data[2410,6]
    y_8 = data[2410,7]
    y_9 = data[2410,8]
    y_10 = data[2410,9]

    equi_before_jump = np.array([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])

    time = Time
    dif_time = np.array([time[int(400/skip)], time[int(400/skip)],
                         time[int(400/skip)],time[int(400/skip)],
                         time[int(400/skip)], time[int(400 / skip)],
                         time[int(400 / skip)], time[int(400 / skip)],
                         time[int(400 / skip)], time[int(400 / skip)]])

    time = Time - Time[2500]
    time_offset = np.array([time[2700], time[2600], time[2580], time[2540], time[2520],
                           time[2516], time[2510], time[2510], time[2504], time[2503]])


    y_1 = data[:2400,0]
    y_2 = data[:2400,1]
    y_3 = data[:2400,2]
    y_4 = data[:2400,3]
    y_5 = data[:2400,4]
    y_6 = data[:2400,5]
    y_7 = data[:2400,6]
    y_8 = data[:2400,7]
    y_9 = data[:2400,8]
    y_10= data[:2400,9]


    baseline_variance = np.var(np.array(#[y_1,
                           [y_1, y_2, y_3,y_4, y_5, y_6, y_7, y_8, y_9, y_10]),axis = 1)



    return after_jump, dif_time, time_offset, equi_before_jump, baseline_variance

def data_slices_decay_new(data, Time, skip):

    data = data.swapaxes(0,1)

    y_1 = data[12510:14510:int(200/skip),0]
    y_2 = data[12510:14510:int(200/skip),1]
    y_3 = data[12510:15510:int(300/skip),2]
    y_4 = data[12510:15510:int(300/skip),3]
    y_5 = data[12510:15510:int(300/skip),4]
    y_6 = data[12510:15510:int(300/skip),5]
    y_7 = data[12510:15510:int(300/skip),6]
    y_8 = data[12510:16510:int(400/skip),7]
    y_9 = data[12510:16510:int(400/skip),8]
    y_10 = data[12510:17510:int(500/skip),9]
    before_jump = np.array([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])

    y_1 = data[12099,0]
    y_2 = data[12099,1]
    y_3 = data[12099,2]
    y_4 = data[12099,3]
    y_5 = data[12099,4]
    y_6 = data[12099,5]
    y_7 = data[12099,6]
    y_8 = data[12099,7]
    y_9 = data[12099,8]
    y_10 = data[12099,9]
    after_jump = np.array([y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10])

    time = Time
    dif_time_dec = np.array([time[int(200/skip)], time[int(200/skip)], time[int(300/skip)], time[int(300/skip)],
                                  time[int(300/skip)],time[int(300 / skip)],time[int(300 / skip)],
                                  time[int(400 / skip)], time[int(400 / skip)], time[int(500 / skip)]])

    time = Time - Time[12500]
    time_offset_dec = np.array([time[12510], time[12510], time[12510], time[12510], time[12510],
                           time[12510], time[12510], time[12510], time[12510], time[12510]])
    return before_jump, dif_time_dec, time_offset_dec, after_jump

def get_command_line_args(sys):

    if len(sys.argv) != 2:
        print(sys.argv)
        print('Invalid Numbers of Arguments. Script will be terminated.')
        return
    else:
        N_channel = int(sys.argv[1])
        print("N_channels: "+str(N_channel))
    return N_channel
def general_info_print_out():
    print("working in: "+os.getcwd())
    localtime = time.asctime(time.localtime(time.time()))
    print(localtime)

def set_instrumenta_noise_standard_deviation():
    singel_std = 0.2
    std = 1.0
    print("std: " + str(std))
    return std, singel_std
def load_the_data(N_channel):
    data = np.load("data"+"/current"+str(N_channel)+".npy")
    Time = np.load("data"+"/Time.npy")
    ligand_conc = np.loadtxt("data"+"/ligand_conc.txt")
    ligand_conc_decay = np.loadtxt("data"+"/ligand_conc_decay.txt")
    return data, Time, ligand_conc, ligand_conc_decay

def define_sampler_params():
    sampling_iter = 9000
    warmingUp = 7000
    chains = 4

    return sampling_iter, warmingUp, chains



def main():
    #setts maximal number of CPUs used on the node
    os.environ["STAN_NUM_THREADS"] = "128"
    general_info_print_out()
    N_channel = get_command_line_args(sys)

    std, single_std = set_instrumenta_noise_standard_deviation()


    data, Time, ligand_conc, ligand_conc_decay = load_the_data(N_channel)
    #skip variable controls how much the actual skipping variabel of the
    # numpy arrays reduced the higher the skip the more data points
    skip = 50.0
    print("skip: "+str(skip))
    data_start, dif_time,     time_of_set_arr, equi_before_jump, baseline_variance = data_slices_beg_new(data, Time, skip)
    data_dec,   dif_time_dec, time_of_set_dec, equi_after_jump  = data_slices_decay_new(data, Time, skip)


    ###########hold_out data dummy since actually the trainings data is used
    data_hold_out, Time_hold_out , ligand_conc, ligand_conc_decay =  load_the_data(N_channel)

    holdout_data_dec, dif_time_dec, time_offset_ignore \
            , hold_equi_after_jump = data_slices_decay_new(data_hold_out, Time, skip)

    holdout_data_start, dif_time, time_offset_ignore, \
        hold_equi_before_jump, baseline_variance_hold = data_slices_beg_new(data_hold_out, Time, skip)



    set_hold_out_start = holdout_data_start
    set_hold_out_equi_before_jump = hold_equi_before_jump
    set_hold_out_decay = holdout_data_dec
    set_hold_out_equi_after = hold_equi_after_jump

    print(time_of_set_dec)
    print(dif_time)
    print(dif_time_dec)
    print("data_start" + str(data_start))
    print("data_dec" + str(data_dec))
    print("equi_before_jump",equi_before_jump)
    print("data_dec"+str(data_dec))
    print("equi_after_jump", equi_after_jump)

    print("dif_time"+str(dif_time))
    print("time_off_begin", time_of_set_arr)
    print("dif_time_dec" + str(dif_time_dec))
    print("time_off_dec", time_of_set_dec)


    N_free_param = 6
    Ion_channels = N_channel
    var_hat_mean = np.mean(baseline_variance)
    #Standard error of the arithmetic mean of the variance
    var_hat_std = np.std(baseline_variance)/np.sqrt(baseline_variance.size)


    Sampling_data_param = {#prior parameters for the instrumental noise
                            "var_open_hat" : np.power(single_std,2),
                            "var_hat_mean": var_hat_mean,#np.power(std,2),
                           "baseline_variance_std": var_hat_std,
                            #data
                            "y_start": data_start,
                           "y_equi_before_jump": equi_before_jump,
                           "y_dec": data_dec,
                           "y_equi_after_jump": equi_after_jump,
                           # time parameters
                           "dif_time": dif_time,
                           "dif_time_dec": dif_time_dec,
                           "off_set_time_arr": time_of_set_arr,
                           "time_off_set_dec": time_of_set_dec,
                           # holdout data
                           "y_equi_before_jump_hold": set_hold_out_equi_before_jump,
                           "y_start_hold": set_hold_out_start,
                           "y_dec_hold": set_hold_out_decay,
                           "y_equi_after_jump_hold": set_hold_out_equi_after,

                           "N_cross_vali": set_hold_out_start.shape[0],
                           "N_data": [len(data_start[0, :]), len(data_dec[0, :])],
                           "N_traces": 4,
                           "N_conc": 10,
                           "N_ion_ch": Ion_channels,

                           "M_states": 4,
                           "N_free_para": N_free_param,
                           "N_open_states": 1,

                           "ligand_conc": ligand_conc,
                           "ligand_conc_decay": ligand_conc_decay


                           }
    print(os.getcwd())
    sampling_iter , warmingUp ,chains = define_sampler_params()
    sampling_iter = 20
    warmingUp = 10
    statistical_model = "KF_CCCO.pic"

    prog_time_start = time.time()
    samples_posterior, model= create_model_and_fit(Sampling_data_param,
                              statistical_model,
                              sampling_iter, warmingUp,
                              chains)

    time_prog_delta = time.time() - prog_time_start
    execution_time = datetime.timedelta(seconds=time_prog_delta).total_seconds()
    print("execution time: "+str(execution_time))
    print(samples_posterior)
    print(os.getcwd())
    save(samples_posterior, data_start, data_dec, holdout_data_start,holdout_data_dec , N_free_param, execution_time)





if __name__ == "__main__":
    main()


