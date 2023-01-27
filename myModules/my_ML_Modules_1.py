# Import of libraries/modules
### -------------------------------------------
import wfdb
from wfdb import processing
### ------------------------------------------- 
import scipy
from scipy.signal import butter, lfilter, filtfilt
### -------------------------------------------
import matplotlib.pyplot as plt
### -------------------------------------------
import numpy as np
### -------------------------------------------
import pandas as pd
### -------------------------------------------
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
### -------------------------------------------
import functools
### -------------------------------------------
import seaborn as sns
### -------------------------------------------
import joblib



# Some Custom Function
def time2Num(time, fs):
    num = time * fs
    return int(num)


def num2Time(num, fs):
    time = num / fs
    return time


def secs2minutes(time):
    return time / 60


def minutes2secs(time):
    return time * 60


def secs2hours(time):
    return time / 3600


def hours2secs(time):
    return time * 3600


def iqr_remove_outlier(x, lower, upper):
    if (x < lower):
        return lower
    elif (x > upper): 
        return upper
    else:
        return x


def correlationTest(signal_1, signal_2, plot=True):
    # Inspect by scatter plot
    if plot: 
        plt.scatter(signal_1, signal_2)
    # Covariance
    covariance = np.cov(signal_1, signal_2)
    print(covariance)
    # calculate Pearson's correlation - 0 is no correlation -1 or 1 is highly correlated
    corr, _ = scipy.stats.pearsonr(signal_1, signal_2)
    print('Pearsons correlation: %.3f' % corr)
    # calculate spearman's correlation - 0 is no correlation -1 or 1 is highly correlated
    corr, _ = scipy.stats.spearmanr(signal_1, signal_2)
    print('Spearmans correlation: %.3f' % corr)


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    
    N = sig.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(sig, color='#3979f0', label='Signal')
    ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x', 
                 color='#8b0000', label='Peak', markersize=12)
    ax_right.plot(np.arange(N), hrs, label='Heart rate', color='m', linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    ax_right.set_ylabel('Heart rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def peaks_rr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.hr.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    
    N = sig.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(sig, color='#3979f0', label='Signal')
    ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x', 
                 color='#8b0000', label='Peak', markersize=12)
    ax_right.plot(np.arange(N), hrs, label='Repiration rate', color='m', linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('RESP (NU)', color='#3979f0')
    ax_right.set_ylabel('Repiration rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()


def read_Data(filename, startNum, endNum):
    signal = wfdb.rdsamp(filename, sampfrom=startNum, sampto=endNum)
    startTime_seconds = startNum/signal[1]['fs']
    endTime_seconds = endNum/signal[1]['fs']
    return signal, startTime_seconds, endTime_seconds


def cal_iqr(signal, hiPerc, loPerc):
    hiPerc_val, loPerc_val = np.percentile(signal, [hiPerc, loPerc])
    iqr = hiPerc_val - loPerc_val
    print(f"{hiPerc}th percentile: {hiPerc_val}, {loPerc}th percentile: {loPerc_val}, IQR: {iqr}")
    return iqr, hiPerc_val, loPerc_val


def iqr_smooth(signal, hiPerc, loPerc, cutoff_factor=0.5):
    # calculate the outlier cutoff
    iqr, hiPerc_val, loPerc_val = cal_iqr(signal, hiPerc=hiPerc, loPerc=loPerc)
    cutoff = iqr * cutoff_factor
    lower, upper = loPerc_val - cutoff, hiPerc_val + cutoff
    # identify outliers
    outliers = [x for x in signal if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    output = map(functools.partial(iqr_remove_outlier, lower=lower, upper=upper), signal)
    output = np.fromiter(output, dtype=np.float64)
    print(f"Data Shape: {output.shape}")
    return output


def norm_signal(signal, max=1, min=-1):
    scaler = MinMaxScaler(feature_range=(min,max))
    output = scaler.fit_transform(signal.reshape((-1,1)))
    return output


def cal_heartrate(signal, fs):
    qrs_inds = processing.qrs.gqrs_detect(sig=signal.reshape(signal.shape[0]), fs=fs)
    hrs = processing.hr.compute_hr(sig_len=signal.shape[0], qrs_inds=qrs_inds, fs=fs)
    return hrs


def cal_resprate(signal, fs):
    peaks_inds = processing.peaks.find_local_peaks(sig=signal.reshape(signal.shape[0]), radius=fs)
    rrs = processing.hr.compute_hr(sig_len=signal.shape[0], qrs_inds=peaks_inds, fs=fs)
    return rrs


def data_fixNan(signal):
    output = pd.DataFrame(signal).fillna(0).to_numpy().reshape(signal.shape[0])
    return output


def data_resample(signal, sig_len):
    output = scipy.signal.resample(signal, sig_len)
    return output


def train_lr_model(X, y, test_size = 0.25, **kwargs):
    
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))

    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    lr_model = LinearRegression().fit(X_train, y_train)
    r_sq = lr_model.score(X_train, y_train)
    print(f"Coefficient of determination: {r_sq}")
    print(f"Intercept: {lr_model.intercept_}")
    print(f"Coefficients: {lr_model.coef_}")
    y_test_predict = lr_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_predict)
    mse = mean_squared_error(y_test, y_test_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return lr_model


def cascade_train_lr_model(model, X, y, test_size = 0.25, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    model.fit(X_train, y_train)
    r_sq = model.score(X_train, y_train)
    print(f"Coefficient of determination: {r_sq}")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    y_test_predict = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_test_predict)
    mse = mean_squared_error(y_test, y_test_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return model


def train_pr_model(X, y, degree=2, test_size = 0.25, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = transformer.fit_transform(X_train)
    X_test_poly = transformer.fit_transform(X_test)
    pr_model = LinearRegression().fit(X_train_poly, y_train)
    r_sq = pr_model.score(X_train_poly, y_train)
    print(f"Coefficient of determination: {r_sq}")
    print(f"Intercept: {pr_model.intercept_}")
    print(f"Coefficients: {pr_model.coef_}")
    y_test_predict = pr_model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_test_predict)
    mse = mean_squared_error(y_test, y_test_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return pr_model


def cascade_train_pr_model(model, X, y, degree=2, test_size = 0.25, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = transformer.fit_transform(X_train)
    X_test_poly = transformer.fit_transform(X_test)
    model.fit(X_train_poly, y_train)
    r_sq = model.score(X_train_poly, y_train)
    print(f"Coefficient of determination: {r_sq}")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    y_test_predict = model.predict(X_test_poly)
    mae = mean_absolute_error(y_test, y_test_predict)
    mse = mean_squared_error(y_test, y_test_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return model


def train_svr_model(X, y, test_size = 0.25, scaler='MinMax', param_C=1, param_gamma=0.1, param_degree=2, param_epsilon=0.1, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    svr_rbf = SVR(kernel='rbf', C=param_C, gamma=param_gamma, epsilon=param_epsilon)
    svr_lin = SVR(kernel='linear', C=param_C)
    svr_poly = SVR(kernel='poly', C=param_C, degree=param_degree, epsilon=param_epsilon)

    # Train models
    svr_rbf.fit(X_train_l, y_train_p.ravel())
    svr_lin.fit(X_train_l, y_train_p.ravel())
    svr_poly.fit(X_train_l, y_train_p.ravel())

    r_sq_rbf = svr_rbf.score(X_train_l, y_train_p)
    r_sq_lin = svr_lin.score(X_train_l, y_train_p)
    r_sq_poly = svr_poly.score(X_train_l, y_train_p)
    print("SVR Radial Basis Function (RBF)")
    print(f"Coefficient of determination: {r_sq_rbf}")
    print(f"Intercept: {svr_rbf.intercept_}")
    print("SVR Linear")
    print(f"Coefficient of determination: {r_sq_lin}")
    print(f"Intercept: {svr_lin.intercept_}")
    print("SVR Polynomial")
    print(f"Coefficient of determination: {r_sq_poly}")
    print(f"Intercept: {svr_poly.intercept_}")

    y_test_p_predict_rbf = svr_rbf.predict(X_test_l)
    y_test_p_predict_lin = svr_lin.predict(X_test_l)
    y_test_p_predict_poly = svr_poly.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_rbf = MinMax_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))
        y_test_predict_lin = MinMax_y.inverse_transform(y_test_p_predict_lin.reshape(-1,1))
        y_test_predict_poly = MinMax_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    else:
        y_test_predict_rbf = StdS_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))
        y_test_predict_lin = StdS_y.inverse_transform(y_test_p_predict_lin.reshape(-1,1))
        y_test_predict_poly = StdS_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    
    print("SVR Radial Basis Function (RBF)")
    mae = mean_absolute_error(y_test, y_test_predict_rbf)
    mse = mean_squared_error(y_test, y_test_predict_rbf)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    print("SVR Linear")
    mae = mean_absolute_error(y_test, y_test_predict_lin)
    mse = mean_squared_error(y_test, y_test_predict_lin)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    print("SVR Polynomial")
    mae = mean_absolute_error(y_test, y_test_predict_poly)
    mse = mean_squared_error(y_test, y_test_predict_poly)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return svr_rbf, svr_lin, svr_poly


def train_svr_rbf_model(X, y, test_size = 0.25, scaler='MinMax', param_C=1, param_gamma=0.1, param_epsilon=0.1, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    svr_rbf = SVR(kernel='rbf', C=param_C, gamma=param_gamma, epsilon=param_epsilon)

    # Train models
    svr_rbf.fit(X_train_l, y_train_p.ravel())

    r_sq_rbf = svr_rbf.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_rbf}")
    print(f"Intercept: {svr_rbf.intercept_}")

    y_test_p_predict_rbf = svr_rbf.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_rbf = MinMax_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))
    else:
        y_test_predict_rbf = StdS_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))

    mae = mean_absolute_error(y_test, y_test_predict_rbf)
    mse = mean_squared_error(y_test, y_test_predict_rbf)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return svr_rbf


def cascade_train_svr_rbf_model(model, X, y, test_size = 0.25, scaler='MinMax', **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    # svr_rbf = SVR(kernel='rbf', C=param_C, gamma=param_gamma, epsilon=param_epsilon)

    # Train models
    model.fit(X_train_l, y_train_p.ravel())

    r_sq_rbf = model.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_rbf}")
    print(f"Intercept: {model.intercept_}")

    y_test_p_predict_rbf = model.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_rbf = MinMax_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))
    else:
        y_test_predict_rbf = StdS_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))

    mae = mean_absolute_error(y_test, y_test_predict_rbf)
    mse = mean_squared_error(y_test, y_test_predict_rbf)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return model


def train_svr_lin_model(X, y, test_size = 0.25, scaler='MinMax', param_C=1, param_gamma='auto', **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    svr_lin = SVR(kernel='linear', C=param_C, gamma=param_gamma)

    # Train models
    svr_lin.fit(X_train_l, y_train_p.ravel())

    r_sq_lin = svr_lin.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_lin}")
    print(f"Intercept: {svr_lin.intercept_}")

    y_test_p_predict_lin = svr_lin.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_lin = MinMax_y.inverse_transform(y_test_p_predict_lin.reshape(-1,1))
    else:
        y_test_predict_lin = StdS_y.inverse_transform(y_test_p_predict_lin.reshape(-1,1))
    mae = mean_absolute_error(y_test, y_test_predict_lin)
    mse = mean_squared_error(y_test, y_test_predict_lin)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return svr_lin


def cascade_train_svr_lin_model(model, X, y, test_size = 0.25, scaler='MinMax', **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    # svr_lin = SVR(kernel='linear', C=param_C, gamma=param_gamma)

    # Train models
    model.fit(X_train_l, y_train_p.ravel())

    r_sq_lin = model.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_lin}")
    print(f"Intercept: {model.intercept_}")

    y_test_p_predict_lin = model.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_lin = MinMax_y.inverse_transform(y_test_p_predict_lin.reshape(-1,1))
    else:
        y_test_predict_lin = StdS_y.inverse_transform(y_test_p_predict_lin.reshape(-1,1))
    mae = mean_absolute_error(y_test, y_test_predict_lin)
    mse = mean_squared_error(y_test, y_test_predict_lin)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return model


def train_svr_poly_model(X, y, test_size = 0.25, scaler='MinMax', param_C=1, param_gamma='auto', param_degree=2, param_epsilon=0.1, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    svr_poly = SVR(kernel='poly', C=param_C, gamma=param_gamma, epsilon=param_epsilon, degree=param_degree)

    # Train models
    svr_poly.fit(X_train_l, y_train_p.ravel())

    r_sq_poly = svr_poly.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_poly}")
    print(f"Intercept: {svr_poly.intercept_}")

    y_test_p_predict_poly = svr_poly.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_poly = MinMax_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    else:
        y_test_predict_y_test_predict_polylin = StdS_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    mae = mean_absolute_error(y_test, y_test_predict_poly)
    mse = mean_squared_error(y_test, y_test_predict_poly)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return svr_poly


def cascade_train_svr_poly_model(model, X, y, test_size = 0.25, scaler='MinMax', **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    # svr_poly = SVR(kernel='poly', C=param_C, gamma=param_gamma, epsilon=param_epsilon, degree=param_degree)

    # Train models
    model.fit(X_train_l, y_train_p.ravel())

    r_sq_poly = model.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_poly}")
    print(f"Intercept: {model.intercept_}")

    y_test_p_predict_poly = model.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_poly = MinMax_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    else:
        y_test_predict_y_test_predict_polylin = StdS_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    mae = mean_absolute_error(y_test, y_test_predict_poly)
    mse = mean_squared_error(y_test, y_test_predict_poly)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return model
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    svr_poly = SVR(kernel='poly', C=param_C, gamma=param_gamma, epsilon=param_epsilon, degree=param_degree)

    # Train models
    svr_poly.fit(X_train_l, y_train_p.ravel())

    r_sq_poly = svr_poly.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_poly}")
    print(f"Intercept: {svr_poly.intercept_}")

    y_test_p_predict_poly = svr_poly.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_poly = MinMax_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    else:
        y_test_predict_y_test_predict_polylin = StdS_y.inverse_transform(y_test_p_predict_poly.reshape(-1,1))
    mae = mean_absolute_error(y_test, y_test_predict_poly)
    mse = mean_squared_error(y_test, y_test_predict_poly)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return svr_poly
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))
    
    if 'seed' in kwargs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=kwargs['seed'])
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)
    
    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_train_l = MinMax_X.fit_transform(X_train)
        y_train_p = MinMax_y.fit_transform(y_train)
        X_test_l = MinMax_X.fit_transform(X_test)
        y_test_p = MinMax_y.fit_transform(y_test)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_train_l = StdS_X.fit_transform(X_train)
        y_train_p = StdS_y.fit_transform(y_train)
        X_test_l = StdS_X.fit_transform(X_test)
        y_test_p = StdS_y.fit_transform(y_test)
    
    # Create models 
    svr_rbf = SVR(kernel='rbf', C=param_C, gamma=param_gamma)

    # Train models
    svr_rbf.fit(X_train_l, y_train_p.ravel())

    r_sq_rbf = svr_rbf.score(X_train_l, y_train_p)
    print(f"Coefficient of determination: {r_sq_rbf}")
    print(f"Intercept: {svr_rbf.intercept_}")

    y_test_p_predict_rbf = svr_rbf.predict(X_test_l)

    if scaler == 'MinMax':
        y_test_predict_rbf = MinMax_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))
    else:
        y_test_predict_rbf = StdS_y.inverse_transform(y_test_p_predict_rbf.reshape(-1,1))
    
    mae = mean_absolute_error(y_test, y_test_predict_rbf)
    mse = mean_squared_error(y_test, y_test_predict_rbf)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    return svr_rbf


def training_pipeline(start_time_secs, end_time_secs, signal_ecg_0, signal_resp_0, ECG_dataset, RESP_dataset):
    ECG_startNum = time2Num(start_time_secs, signal_ecg_0[1]['fs'])
    ECG_endNum = time2Num(end_time_secs, signal_ecg_0[1]['fs'])
    print(f'start: {ECG_startNum}, end: {ECG_endNum}')
    RESP_startNum = time2Num(start_time_secs, signal_resp_0[1]['fs'])
    RESP_endNum = time2Num(end_time_secs, signal_resp_0[1]['fs'])
    print(f'start: {RESP_startNum}, end: {RESP_endNum}')
    signal_ECG, signal_ECG_startTime_secs, signal_ECG_endTime_secs = read_Data(ECG_dataset, startNum=ECG_startNum, endNum=ECG_endNum)
    signal_RESP, signal_RESP_startTime_secs, signal_RESP_endTime_secs = read_Data(RESP_dataset, startNum=RESP_startNum, endNum=RESP_endNum)
    # print(signal_ECG[0].shape)
    signal_ECG_1 = iqr_smooth(signal=signal_ECG[0],hiPerc=90,loPerc=10, cutoff_factor=1)
    # print(signal_RESP[0].shape)
    signal_RESP_1 = iqr_smooth(signal=signal_RESP[0],hiPerc=90,loPerc=10, cutoff_factor=1)
    signal_ECG_2 = norm_signal(signal=signal_ECG_1)
    signal_RESP_2 = norm_signal(signal=signal_RESP_1)
    hrs_0 = cal_heartrate(signal=signal_ECG_2, fs=signal_ECG[1]['fs'])
    rrs_0 = cal_resprate(signal=signal_RESP_2, fs=signal_RESP[1]['fs'])
    if (rrs_0.shape[0] < hrs_0.shape[0]):
        hrs_1 = data_resample(data_fixNan(hrs_0), rrs_0.shape[0])
        rrs_1 = data_fixNan(rrs_0)
    else:
        rrs_1 = data_resample(data_fixNan(rrs_0), hrs_0.shape[0])
        hrs_1 = data_fixNan(hrs_0)
    correlationTest(rrs_1[:],hrs_1[:], plot=False)
    print("---------------------------------------------")
    print("--- Linear Regression ---")
    lr_model_1 = train_lr_model(X=rrs_1, y=hrs_1)
    print("---------------------------------------------")
    print("--- Polynomial Regression ---")
    pr_model_1 = train_pr_model(X=rrs_1, y=hrs_1, degree=6)
    print("---------------------------------------------")
    print("--- SVR Radius Basis Function---")
    svr_rbf_1 = train_svr_rbf_model(X=rrs_1, y=hrs_1, param_C=100, param_gamma=0.1, seed=42)
    # print("--- SVR Linear ---")
    # svr_lin_1 = train_svr_lin_model(X=rrs_1, y=hrs_1, param_C=1000, param_gamma='auto', seed=42)
    # print("--- SVR Polynomial ---")
    # svr_poly_1 = train_svr_poly_model(X=rrs_1, y=hrs_1, param_C=1000, param_gamma='auto', param_degree=6, seed=42)
    return lr_model_1, pr_model_1, svr_rbf_1


def cascade_training_pipeline(lr_model, pr_model, svr_rbf_model, start_time_secs, end_time_secs, signal_ecg_0, signal_resp_0, ECG_dataset, RESP_dataset):
    ECG_startNum = time2Num(start_time_secs, signal_ecg_0[1]['fs'])
    ECG_endNum = time2Num(end_time_secs, signal_ecg_0[1]['fs'])
    print(f'start: {ECG_startNum}, end: {ECG_endNum}')
    RESP_startNum = time2Num(start_time_secs, signal_resp_0[1]['fs'])
    RESP_endNum = time2Num(end_time_secs, signal_resp_0[1]['fs'])
    print(f'start: {RESP_startNum}, end: {RESP_endNum}')
    signal_ECG, signal_ECG_startTime_secs, signal_ECG_endTime_secs = read_Data(ECG_dataset, startNum=ECG_startNum, endNum=ECG_endNum)
    signal_RESP, signal_RESP_startTime_secs, signal_RESP_endTime_secs = read_Data(RESP_dataset, startNum=RESP_startNum, endNum=RESP_endNum)
    # print(signal_ECG[0].shape)
    signal_ECG_1 = iqr_smooth(signal=signal_ECG[0],hiPerc=90,loPerc=10, cutoff_factor=1)
    # print(signal_RESP[0].shape)
    signal_RESP_1 = iqr_smooth(signal=signal_RESP[0],hiPerc=90,loPerc=10, cutoff_factor=1)
    signal_ECG_2 = norm_signal(signal=signal_ECG_1)
    signal_RESP_2 = norm_signal(signal=signal_RESP_1)
    hrs_0 = cal_heartrate(signal=signal_ECG_2, fs=signal_ECG[1]['fs'])
    rrs_0 = cal_resprate(signal=signal_RESP_2, fs=signal_RESP[1]['fs'])
    if (rrs_0.shape[0] < hrs_0.shape[0]):
        hrs_1 = data_resample(data_fixNan(hrs_0), rrs_0.shape[0])
        rrs_1 = data_fixNan(rrs_0)
    else:
        rrs_1 = data_resample(data_fixNan(rrs_0), hrs_0.shape[0])
        hrs_1 = data_fixNan(hrs_0)
    correlationTest(rrs_1[:],hrs_1[:], plot=False)
    print("---------------------------------------------")
    print("--- Linear Regression ---")
    lr_model = cascade_train_lr_model(model=lr_model, X=rrs_1, y=hrs_1)
    print("---------------------------------------------")
    print("--- Polynomial Regression ---")
    pr_model = cascade_train_pr_model(model=pr_model, X=rrs_1, y=hrs_1, degree=6)
    print("---------------------------------------------")
    print("--- SVR Radius Basis Function---")
    svr_rbf_model = cascade_train_svr_rbf_model(model=svr_rbf_model, X=rrs_1, y=hrs_1, param_C=100, param_gamma=0.1)
    # print("--- SVR Linear ---")
    # svr_lin_model = cascade_train_svr_lin_model(model=svr_lin_model, X=rrs_1, y=hrs_1, param_C=1000, param_gamma='auto')
    # print("--- SVR Polynomial ---")
    # svr_poly_model = cascade_train_svr_poly_model(model=svr_poly_model, X=rrs_1, y=hrs_1, param_C=1000, param_gamma='auto', param_degree=6)
    return lr_model, pr_model, svr_rbf_model


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def test_lr_model(model, X, y, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))

    y_predict = model.predict(X)
    mae = mean_absolute_error(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return y_predict


def average_lr_model_predict(X, y):
    step = 0.25
    startHrs = 10
    endHrs = 12

    filename = f"../models/infant_1/lr_model_step{step}_startHrs{startHrs}_endHrs{endHrs}.sav"
    print(filename)
    load_lr_models = joblib.load(filename)
    y_predict = test_lr_model(model=load_lr_models, X=X, y=y)

    for infantNum in range(2, 9, 1):
        filename = f"../models/infant_{infantNum}/lr_model_step{step}_startHrs{startHrs}_endHrs{endHrs}.sav"
        print(filename)
        load_lr_models = joblib.load(filename)
        y_predict_temp = test_lr_model(model=load_lr_models, X=X, y=y)
        y_predict += y_predict_temp

    y_predict = y_predict/8
    print("-------------------------------------------------------------------------------------------")
    mae = mean_absolute_error(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return y_predict


def data_bin(start_time_secs, end_time_secs, signal_ecg_0, signal_resp_0, ECG_dataset, RESP_dataset):
    ECG_startNum = time2Num(start_time_secs, signal_ecg_0[1]['fs'])
    ECG_endNum = time2Num(end_time_secs, signal_ecg_0[1]['fs'])
    print(f'start: {ECG_startNum}, end: {ECG_endNum}')
    RESP_startNum = time2Num(start_time_secs, signal_resp_0[1]['fs'])
    RESP_endNum = time2Num(end_time_secs, signal_resp_0[1]['fs'])
    print(f'start: {RESP_startNum}, end: {RESP_endNum}')
    signal_ECG, signal_ECG_startTime_secs, signal_ECG_endTime_secs = read_Data(ECG_dataset, startNum=ECG_startNum, endNum=ECG_endNum)
    signal_RESP, signal_RESP_startTime_secs, signal_RESP_endTime_secs = read_Data(RESP_dataset, startNum=RESP_startNum, endNum=RESP_endNum)
    # print(signal_ECG[0].shape)
    signal_ECG_1 = iqr_smooth(signal=signal_ECG[0],hiPerc=90,loPerc=10, cutoff_factor=1)
    # print(signal_RESP[0].shape)
    signal_RESP_1 = iqr_smooth(signal=signal_RESP[0],hiPerc=90,loPerc=10, cutoff_factor=1)
    signal_ECG_2 = norm_signal(signal=signal_ECG_1)
    signal_RESP_2 = norm_signal(signal=signal_RESP_1)
    hrs_0 = cal_heartrate(signal=signal_ECG_2, fs=signal_ECG[1]['fs'])
    rrs_0 = cal_resprate(signal=signal_RESP_2, fs=signal_RESP[1]['fs'])
    if (rrs_0.shape[0] < hrs_0.shape[0]):
        hrs_1 = data_resample(data_fixNan(hrs_0), rrs_0.shape[0])
        rrs_1 = data_fixNan(rrs_0)
    else:
        rrs_1 = data_resample(data_fixNan(rrs_0), hrs_0.shape[0])
        hrs_1 = data_fixNan(hrs_0)
    return rrs_1, hrs_1


def test_pr_model(model, X, y, degree=2, **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))

    transformer = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = transformer.fit_transform(X)
    y_predict = model.predict(X_poly)
    mae = mean_absolute_error(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return y_predict


def average_pr_model_predict(X, y):
    step = 0.25
    startHrs = 10
    endHrs = 12
    degree = 6

    filename = f"../models/infant_1/pr_model_step{step}_startHrs{startHrs}_endHrs{endHrs}.sav"
    print(filename)
    load_pr_models = joblib.load(filename)
    y_predict = test_pr_model(model=load_pr_models, X=X, y=y, degree=degree)

    for infantNum in range(2, 9, 1):
        filename = f"../models/infant_{infantNum}/pr_model_step{step}_startHrs{startHrs}_endHrs{endHrs}.sav"
        print(filename)
        load_pr_models = joblib.load(filename)
        y_predict_temp = test_pr_model(model=load_pr_models, X=X, y=y, degree=degree)
        y_predict += y_predict_temp

    y_predict = y_predict/8
    print("-------------------------------------------------------------------------------------------")
    mae = mean_absolute_error(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return y_predict


def test_svr_rbf_model(model, X, y, scaler='MinMax', **kwargs):
    X = X.reshape((-1,1))
    y = y.reshape((-1,1))

    # feature scaling
    if scaler == 'MinMax':
        MinMax_X = MinMaxScaler()
        MinMax_y = MinMaxScaler()
        X_l = MinMax_X.fit_transform(X)
        # y_p = MinMax_y.fit_transform(y)
    else: 
        StdS_X = StandardScaler()
        StdS_y = StandardScaler()
        X_l = StdS_X.fit_transform(X)
        # y_p = StdS_y.fit_transform(y)

    y_p_predict = svr_rbf.predict(X_l)

    if scaler == 'MinMax':
        y_predict = MinMax_y.inverse_transform(y_p_predict.reshape(-1,1))
    else:
        y_predict = StdS_y.inverse_transform(y_p_predict.reshape(-1,1))
    
    mae = mean_absolute_error(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return y_predict


def average_svr_rbf_model_predict(X, y):
    step = 0.25
    startHrs = 10
    endHrs = 12
    degree = 6

    filename = f"../models/infant_1/svr_rbf_model_step{step}_startHrs{startHrs}_endHrs{endHrs}.sav"
    print(filename)
    load_svr_rbf_models = joblib.load(filename)
    y_predict = test_svr_rbf_model(model=load_svr_rbf_models, X=X, y=y)

    for infantNum in range(2, 9, 1):
        filename = f"../models/infant_{infantNum}/svr_rbf_model_step{step}_startHrs{startHrs}_endHrs{endHrs}.sav"
        print(filename)
        load_svr_rbf_models = joblib.load(filename)
        y_predict_temp = test_svr_rbf_model(model=load_svr_rbf_models, X=X, y=y)
        y_predict += y_predict_temp

    y_predict = y_predict/8
    print("-------------------------------------------------------------------------------------------")
    mae = mean_absolute_error(y, y_predict)
    mse = mean_squared_error(y, y_predict)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')
    return y_predict

