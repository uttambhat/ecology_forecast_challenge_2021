#=================================================================
# A cleaned version of other scripts to run with run_phenology.sh
# Same as analyse_fourier.py as on 2021-02-24 1354 hrs
#=================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import date
today = date.today().strftime("%Y-%m-%d")

data_raw = pd.read_csv("targets/phenology-targets_" + str(today) + ".csv",sep=',')
data_raw = data_raw.dropna()
data_raw['datetime']=pd.to_datetime(data_raw['time'])
data_raw['juliandate']=pd.DatetimeIndex(data_raw['time']).to_julian_date()
data_all = data_raw.groupby(data_raw.siteID)

output_data = {}
for site in data_all.groups.keys():
    data = data_all.get_group(site)
    data = data.reset_index()
    data['juliandate_adjusted']=data['juliandate']-data['juliandate'][0]
    data_1 = np.array([list(a) for a in zip(data.juliandate_adjusted,data.gcc_90)])
    
    #=================================================================================
    # The code below is copied aquatics/from analyse_gp.py on 2021-01-2 1334 hours
    # Remove redundancies and make a separate library for common code
    #=================================================================================
    
    #===============================
    # Create training and test sets 
    #===============================
    trainlength = int(len(data_1))
    testlength = len(data_1)#-trainlength
    x_train,y_train = data_1[:trainlength,0],data_1[:trainlength,1]
    x_test,y_test = data_1[-testlength:,0],data_1[-testlength:,1]
    forecast_ahead = np.round(pd.Timestamp.today().to_julian_date()-data['juliandate'].max())
    x_new = np.arange(x_test[-1]+forecast_ahead,x_test[-1]+35+forecast_ahead,1.)
    
    fourier_period = 365.
    max_frequency = 50.
    frequency_list = np.arange(0.,max_frequency+1.,1.)
    
    # Training feature vectors
    Hcos = np.array([np.cos(2.*np.pi*k*x_train/fourier_period) for k in frequency_list]).T
    Hsin = np.array([np.sin(2.*np.pi*k*x_train/fourier_period) for k in frequency_list[1:]]).T
    H_train = np.concatenate((Hcos,Hsin),axis=1)
    Y_train = y_train.reshape(-1,1)
    Y2_train = (y_train**2).reshape(-1,1)
    
    # Test feature vectors
    Hcos = np.array([np.cos(2.*np.pi*k*x_test/fourier_period) for k in frequency_list]).T
    Hsin = np.array([np.sin(2.*np.pi*k*x_test/fourier_period) for k in frequency_list[1:]]).T
    H_test = np.concatenate((Hcos,Hsin),axis=1)
    Y_test = y_test.reshape(-1,1)
    
    # New data prediction feature vectors
    Hcos = np.array([np.cos(2.*np.pi*k*x_new/fourier_period) for k in frequency_list]).T
    Hsin = np.array([np.sin(2.*np.pi*k*x_new/fourier_period) for k in frequency_list[1:]]).T
    H_new = np.concatenate((Hcos,Hsin),axis=1)
    
    #==========================================================================================================================================
    # Regularized Fourier regression fit for Y and Y**2 (to get variance as well)
    # Note: This is same as a Fourier transform (see ref below), but it is easier to regularize
    # (see https://stats.stackexchange.com/questions/249198/from-a-statistical-perspective-fourier-transform-vs-regression-with-fourier-bas
    #==========================================================================================================================================
    Omega = np.diag(np.concatenate(((fourier_period/2.)*(np.pi*frequency_list/(fourier_period/2.))**4,(fourier_period/2.)*(np.pi*frequency_list[1:]/(fourier_period/2.))**4)))
    lambd_list = [1000.]#[1.,10.,100.,1000.,10000.]
    b_reg,b2_reg,Y_pred_reg,Y2_pred_reg,std_pred,Y_new_reg,Y2_new_reg,std_new = {},{},{},{},{},{},{},{}
    for lambd in lambd_list:
        b_reg[lambd] = np.linalg.inv(H_train.T@H_train+lambd*Omega)@(H_train.T)@Y_train
        Y_pred_reg[lambd] = H_test@b_reg[lambd]
        Y_new_reg[lambd] = H_new@b_reg[lambd]
        b2_reg[lambd] = np.linalg.inv(H_train.T@H_train+lambd*Omega)@(H_train.T)@Y2_train
        Y2_pred_reg[lambd] = H_test@b2_reg[lambd]
        Y2_new_reg[lambd] = H_new@b2_reg[lambd]
        std_pred[lambd] = np.sqrt(np.abs(Y2_pred_reg[lambd] - Y_pred_reg[lambd]**2))
        std_new[lambd] = np.sqrt(np.abs(Y2_new_reg[lambd] - Y_new_reg[lambd]**2))
    
    
    # Add formatted output to output_data dictionary
    output_data[site] = []
    for i in range(len(x_new)):
        # mean
        output_data[site].append(str((data.datetime[0]+pd.to_timedelta(x_new[i],unit='d')).date()))
        output_data[site][-1] += ','+site+",2,1,1,mean,"+str(Y_new_reg[lambd][i,0])+'\n'
        # std
        output_data[site].append(str((data.datetime[0]+pd.to_timedelta(x_new[i],unit='d')).date()))
        output_data[site][-1] += ','+site+",2,1,1,sd,"+str(std_new[lambd][i,0])+'\n'
    
    lambd=1000
    
    # Plot all data
    """
    plt.scatter(x_test,y_test,label="Data")
    for lambd in b_reg.keys():
    x = np.concatenate((x_test,x_new))
    y = np.concatenate((Y_pred_reg[lambd][:,0],Y_new_reg[lambd][:,0]))
    std = np.concatenate((std_pred[lambd][:,0],std_new[lambd][:,0]))
    plt.scatter(x,y,label = "Lambda = "+str(lambd) + ", RMSE = "+str(np.sqrt(mean_squared_error(Y_test,Y_pred_reg[lambd]))/np.std(Y_test))[:4])
    plt.fill_between(x,(y-std),(y+std),alpha=0.5,facecolor="orange")
    
    plt.xlabel("t (days)")
    plt.ylabel("gcc_90")
    plt.legend()
    plt.title("Data vs. Prediction")
    plt.show()
    """
    # Just the forecast
    """
    x = x_new
    y = Y_new_reg[lambd][:,0]
    std = std_new[lambd][:,0]
    plt.scatter(x,y,label = "Forecasts at site "+site)
    plt.fill_between(x,(y-std),(y+std),alpha=0.5,facecolor="orange")
    plt.xlabel("t (days)")
    plt.ylabel("gcc_90")
    plt.legend()
    plt.title("Forecasts")
    plt.show()
    """

#outputfile = open("test.csv","w")
outputfile = open("submissions/phenology-"+today+"-Fourier.csv","w")
outputfile.write("time,siteID,obs_flag,forecast,data_assimilation,statistic,gcc_90\n")
for i in range(len(x_new)):
    for site in output_data.keys():
        outputfile.write(output_data[site][2*i])
        outputfile.write(output_data[site][2*i+1])

outputfile.close()

