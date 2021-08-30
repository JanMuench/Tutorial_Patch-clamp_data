
import os
import numpy as np
import pandas as pd

import xarray as xr





def save_data_new(fit, data_start, data_dec,dataStartHold, dataDecHold, N_free_param, execution_time):

        #try:

        #    print("trying the modern version with xarray and arviz")
        #    import arviz as az
        #    import xarray
        #    az.from_pystan(
        #        posterior=fit)

        #except:
        #    print("failed to this in the modern version")
        try:
            stepsize = fit.get_stepsize()
            print("step size" + str(stepsize))[0]
            # by default .get_inv_metric returns a list
            inv_metric = fit.get_inv_metric(as_dict=True)[0]
            init = fit.get_last_position()[0]

            # increment seed by 1

            control = {"stepsize": stepsize,
                   "inv_metric": inv_metric,
                   "adapt_engaged": False
                   }
            np.save("inv_metric_sampler", inv_metric)
            np.save("last_param_position", init)
            np.save("seed", seed)
            np.save("setp_size", stepsize)
        except:
            print("could not save control params")
            pass



        # if not os.path.exists(folder):
        #    os.makedirs(folder)
        # os.chdir(folder)
        print("saving in: " + os.getcwd())

        np.save("data_start", data_start)
        np.save("data_dec", data_dec)
        np.save("data_start_hold", dataStartHold)
        np.save("data_dec_hold", dataDecHold)

        exec_time = np.array(execution_time)
        np.save("execution_time_in_seconds", exec_time)

        #try:
        #    sampling_data = pd.DataFrame(fit.extract(["log_dwell_times", "ratio", ], permuted=True))
        #except:
        #    print("cannot make a pandas data frame")

        #try:
        #    sampling_data.to_csv("sampling_daten")
        #except:
        #    print("could not save fit_data")

        for name in ("param_likelihood_start","ParamLikeliStartHoldout"):
            try:
                param_likelihood = np.array(fit.extract(name, permuted=True)[name])
                param_likelihood = np.swapaxes(param_likelihood, 0, 1)
                print("param_like.shape: "+ param_likelihood.shape)
            except:
                print("param likihood existiert nicht")

            try:
                major_axis = list()
                for i in range(1, 21):
                    major_axis.append(str(i))

                param = xr.DataArray(data=param_likelihood[:, :, :, :],
                                 dims=("N_conc_time_series", "samples_posterior", "data_point", "parameter_likelihood"),
                                 coords={
                                     "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16",
                                                            "64"],
                                     "parameter_likelihood": ["mean", "sigma"]})
                param.to_netcdf(name)
            except:
                print("could not save likelihood")
        for fname in ("param_likelihood_decay", "ParamLikeliDecayHoldout"):
            try:
                param_likelihood_decay = np.array(
                fit.extract(fname, permuted=True)[fname])
                param_likelihood_decay = np.swapaxes(param_likelihood_decay, 0, 1)
                param = xr.DataArray(data=param_likelihood_decay[:, :, :, :],
                                 dims=("N_conc_time_series", "samples_posterior", "data_point", "parameter_likelihood"),
                                 coords={
                                     "N_conc_time_series": ["0.0625", "0.125", "0.25", "0.5", "1", "2", "4", "8", "16",
                                                            "64"],
                                     "parameter_likelihood": ["mean", "sigma"]})
                param.to_netcdf(fname)
            except:
                print("could not save likelihood")

        try:
            backround_sigma = np.array(fit.extract("var_exp", permuted=True)["var_exp"])
            np.save("measurement_sigma", np.array(backround_sigma))
        except:
            print("could save backround noise")
        try:
            N_traces = fit.extract("N_ion_trace", permuted=True)["N_ion_trace"]
            np.save("N_traces", np.array(N_traces))
        except:
            print("N_traces param to fit")

        try:
            hyper_mu_N = fit.extract("hyper_mu_N", permuted=True)["hyper_mu_N"]
            sigma_N = fit.extract("sigma_N", permuted=True)["sigma_N"]
            np.save("hyper_mu_N", hyper_mu_N)
            np.save("sigma_N", sigma_N)
        except:
            pass

        try:
            mu_i = fit.extract("mu_i", permuted=True)["mu_i"]
            sigma_i = fit.extract("sigma_i", permuted=True)["sigma_i"]
            np.save("mu_i", mu_i)
            np.save("sigma_i", sigma_i)
        except:
            pass


        try:
            N_traces = fit.extract("mu_N", permuted=True)["mu_N"]
            np.save("mu_N", np.array(N_traces))
        except:
            print("mu_N param to fit")

        try:
            N_traces = fit.extract("var_N", permuted=True)["var_N"]
            np.save("var_N", np.array(N_traces))
        except:
            print("var_N param to fit")

        try:
            mu_k = fit.extract("mu_k", permuted=True)["mu_k"]
            np.save("mu_k", np.array(mu_k))
            sigma_k = fit.extract("sigma_k", permuted=True)["sigma_k"]
            np.save("sigma_k", np.array(sigma_k))
        except:
            pass



        try:
            open_variance = fit.extract("open_variance", permuted=True)["open_variance"]
            np.save("open_variance", np.array(open_variance))
        except:
            print("could not save open_variance param to fit")

        try:
            lp__ = fit.extract("lp__", permuted=True)["lp__"]
            lp__ = pd.DataFrame(data=lp__)
            lp__.to_csv("lp__")
        except:
            print("lp_ saving doesn t work")

        try:
            latent_time = fit.extract("LATENT_TIME", permuted=True)["LATENT_TIME"]
            np.save("latent_time", np.array(latent_time))
        except:
            print("LATENT TIME doesn t exist")

        try:
            latent_time_decay = fit.extract("LATENT_TIME_DECAY", permuted=True)["LATENT_TIME_DECAY"]
            np.save("latent_time_decay", np.array(latent_time_decay))
        except:
            print("LATENT TIME doesn t exist")

        try:
            occupat_dec = fit.extract("occupat_decay", permuted=True)["occupat_decay"]
            np.save("occupat_dec2", np.array(occupat_dec))
        except:
            print("occupat_decay doesn t exist")

        # mu = fit.extract("mu", permuted = True)["mu"]
        # np.save("mu", np.array(mu))
        try:
            equi_values = fit.extract("equi_values", permuted=True)["equi_values"]
            np.save("equi_values2", np.array(equi_values))
        except:
            print("could not open equi_values")

        try:
            occupat = fit.extract("occupat", permuted=True)["occupat"]
            print(occupat)
            np.save("occupat2", np.array(occupat))
        except:
            print("could not save occupat")

        try:
            log_lik_t = fit.extract("log_lik_t", permuted=True)["log_lik_t"]
            np.save("log_lik_t2", np.array(log_lik_t))
        except:
            print("could not save log_lik_t")

        try:
            log_lik_h = fit.extract("logLikHoldout", permuted=True)["logLikHoldout"]
            np.save("logLikHoldout", np.array(log_lik_h))
        except:
            print("cold not save log_lik_h")

        column_names = list()
        for id in range(1, np.int(N_free_param / 2 + 1)):
            column_names.append("log_dwell_time[" + str(id) + "]")


        theta = fit.extract("log_dwell_times", permuted=True)
        theta = pd.DataFrame(data=theta["log_dwell_times"], columns=column_names)
        #theta.to_csv("log_dwell_times")

        column_names = list()
        for id in range(1, np.int(N_free_param / 2 +1)):

            if id == np.int(N_free_param / 2 ):
                column_names.append("log_dwell_times[" + str(id+1) + "]")
            else :
                column_names.append("ratio[" + str(id) + "]")

        ratio = fit.extract("ratio", permuted=True)
        ratio = pd.DataFrame(data=ratio["ratio"], columns=column_names)
        rate_matrix_params = pd.concat([theta, ratio], axis=1, join='inner')
        rate_matrix_params.to_csv("rate_matrix_params", index=False)
        print(rate_matrix_params.values)
        print(rate_matrix_params.values.shape)
        print(pd.read_csv("rate_matrix_params"))
        try:
            i_single = fit.extract("i_single_channel", permuted=True)["i_single_channel"]
            np.save("i_single", np.array(i_single))
        except:
            print("i_single problems")











def main():
    save_data(bla)
if __name__ == "__main__":
    main()