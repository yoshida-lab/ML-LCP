import os
import platform
import time
import pickle
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import importlib

import copy
import math

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import KFold

from scipy.stats import bootstrap
from scipy.optimize import curve_fit

import datetime
from sklearn.linear_model import LogisticRegression





def fix_seed(seed):
    # Numpy
    np.random.seed(seed)
    # for built-in random
    random.seed(int(seed))
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    
    
    
    
def fun_noninfomativeFeatures(df):
    equal_columns = df.apply(lambda col: col.nunique() == 1)
    equal_column_names = equal_columns[equal_columns].index.tolist()
    notequal_column_names = equal_columns[-equal_columns].index.tolist()
    return notequal_column_names #, equal_column_names

    
    
    
    
class MyScaler():
    def __init__(self, method=['Normalization']):
        self.method = method
    
    class Scaler():
        def __init__(self, method='Normalization'):
            self.method = method
            
        def fun_CalcPlus(self, series):
            tmp_min = series.min()
            second_min = pd.Series(series.unique()).nsmallest(2).iloc[-1]
            if tmp_min > 0:
                return 0
            if tmp_min <= 0:
                return -2*tmp_min + second_min
            
        def fun_Custom(self, x):
            if x >= 1:
                return self.slope_b * x - self.slope_b + self.slope_a
            if x <= -1:
                return self.slope_b * x + self.slope_b - self.slope_a
            else:
                return self.slope_a * x
            
        def fun_InvCustom(self, x):
            if x >= self.slope_a:
                return x/self.slope_b + 1 - self.slope_a/self.slope_b
            if x <= - self.slope_a:
                return x/self.slope_b - 1 + self.slope_a/self.slope_b
            else:
                return x / self.slope_a
        
        def fit_transform(self, df):
            self.SeriesFlg = 0
            if isinstance(df, pd.Series):
                self.SeriesFlg = 1
                df = pd.DataFrame(df)
                
            if self.method=='Normalization':
                self.mean = df.mean()
                self.std = df.std()
                df_scal = (df - self.mean)/self.std
            
            if self.method=='Yeo-Johnson':
                pt = PowerTransformer(method='yeo-johnson', standardize=True)
                tmp_df = pd.DataFrame(pt.fit_transform(df), columns=df.columns, index=df.index)
                self.pt = pt
                df_scal = tmp_df
            
            if self.method=='PlusLog':
                self.plus = df.apply(self.fun_CalcPlus)
                df_scal = np.log(df + self.plus)
            
            if self.method=='MinMax':
                self.max = df.max()
                self.min = df.min()
                df_scal = (df - self.min)/(self.max - self.min)
        
            if self.method=='MinMaxPlusOne':
                self.max = df.max()
                self.min = df.min()
                df_scal = (df - self.min)/(self.max - self.min) + 1
            
            if self.method=='CubeRoot':
                df_scal = np.cbrt(df)
                
            if self.method=='Custom':
                self.slope_a = 2
                self.slope_b = 1/2
                df_scal = df.applymap(self.fun_Custom)
            
            if self.SeriesFlg == 1:
                df_scal = df_scal.squeeze()
                
            return df_scal
        
        
        def transform(self, df):
            self.SeriesFlg = 0
            if isinstance(df, pd.Series):
                self.SeriesFlg = 1
                df = pd.DataFrame(df)
                
            if self.method=='Normalization':
                df_scal = (df - self.mean)/self.std
            
            if self.method=='Yeo-Johnson':
                tmp_df = pd.DataFrame(self.pt.transform(df), columns=df.columns, index=df.index)
                df_scal = tmp_df
            
            if self.method=='PlusLog':
                df_scal = np.log(df + self.plus)
            
            if self.method=='MinMax':
                df_scal = (df - self.min)/(self.max - self.min)
        
            if self.method=='MinMaxPlusOne':
                df_scal = (df - self.min)/(self.max - self.min) + 1
            
            if self.method=='CubeRoot':
                df_scal = np.cbrt(df)
                
            if self.method=='Custom':
                df_scal = df.applymap(self.fun_Custom)
            
            if self.SeriesFlg == 1:
                df_scal = df_scal.squeeze()
                
            return df_scal
        
            
        def inverse(self, df):
            if self.SeriesFlg == 1:
                df = pd.DataFrame(df)
            
            if self.method=='Normalization':
                df_rescal = (df*self.std)+self.mean
            
            if self.method=='Yeo-Johnson':
                df_rescal = pd.DataFrame(self.pt.inverse_transform(df), columns=df.columns, index=df.index)
            
            if self.method=='PlusLog':
                df_rescal = np.exp(df) - self.plus
            
            if self.method=='MinMax':
                df_rescal = df * (self.max - self.min) + self.min
        
            if self.method=='MinMaxPlusOne':
                df_rescal = (df-1) * (self.max - self.min) + self.min
        
            if self.method=='CubeRoot':
                df_rescal = df**3
                
            if self.method=='Custom':
                df_rescal = df.applymap(self.fun_InvCustom)
            
            if self.SeriesFlg == 1:
                df_rescal = df_rescal.squeeze()
                
            return df_rescal

    def fit_transform(self, df):
        self.scaler = []
        for m in self.method:
            scaler = self.Scaler(method=m)
            df = scaler.fit_transform(df)
            self.scaler.append(scaler)
        return df
    
    def transform(self, df):
        for i, m in enumerate(self.method):
            scaler = self.scaler[i]
            df = scaler.transform(df)
            self.scaler.append(scaler)
        return df
    
    def inverse(self, df):
        for m, s in zip(list(reversed(self.method)), list(reversed(self.scaler))):
            df = s.inverse(df)
        return df
    
    
    
    
    
def fun_pltHist(df, title, showflg=False, savename=None):
    # plt.figure()
    if isinstance(df, pd.Series):
        _figsize = np.int8(np.ceil(np.sqrt(1)))*3
    else:
        _figsize = np.int8(np.ceil(np.sqrt(df.shape[1])))*3
    df.hist(figsize=(_figsize,_figsize))
    plt.suptitle(title)
    plt.tight_layout(rect=[0,0,1,0.98])
    if savename is not None:
        plt.savefig(savename)
    if showflg:
        plt.show()
    plt.close()
    
    
    
    
    
def plot_scatter(y_obs_list, 
                 y_prd_list, 
                 title_list, 
                 plt_row, 
                 plt_col, 
                 position_list, 
                 col_list,
                 alpha_list,
                 fig_size, 
                 save_name, 
                 title, 
                 show_flg=True):
    fig = plt.figure(figsize=fig_size)

    for i_plt in range(len(position_list)):
        ax = fig.add_subplot(plt_row, plt_col, position_list[i_plt], 
                             title=title_list[i_plt], 
                             xlabel='Observation', 
                             ylabel='Prediction')
        ax.scatter(y_obs_list[i_plt], y_prd_list[i_plt], color=col_list[i_plt], alpha=alpha_list[i_plt])
        xy_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
        xy_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.axis('equal')
        ax.axis('square')
        ax.set_xlim([xy_min, xy_max])
        ax.set_ylim([xy_min, xy_max])
        ax.grid(color='gray', linestyle='dotted', linewidth=1, alpha=0.5)
        
        try:
            ax.text(0.03, 0.93, 'Corr : '+str(round(np.corrcoef(y_prd_list[i_plt], y_obs_list[i_plt])[0,1], 4)), size=15, transform=ax.transAxes)
            ax.text(0.03, 0.87, 'RMSE : '+str(round(mean_squared_error(y_obs_list[i_plt], y_prd_list[i_plt], squared=False), 4)), size=15, transform=ax.transAxes)
            ax.text(0.03, 0.81, 'MAE : '+str(round(mean_absolute_error(y_obs_list[i_plt], y_prd_list[i_plt]), 4)), size=15, transform=ax.transAxes)
            ax.text(0.03, 0.75, 'R2 : '+str(round(r2_score(y_obs_list[i_plt], y_prd_list[i_plt]), 4)), size=15, transform=ax.transAxes)
        except:
            ax.text(0.03, 0.93, '', size=15, transform=ax.transAxes)
        _ = ax.plot([-300000, 300000], [-300000, 300000], color='gray', linewidth=0.5)

    fig.tight_layout(rect=[0,0,1,0.90])
    
    plt.suptitle(title,fontsize=20)

    fig.savefig(save_name)
    if show_flg==False:
        plt.close(fig)
        
        
        
        
        
        
def MyTime(start_time, i, num_iterations):
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y/%m/%d/%H:%M:%S")
    
    # Calculate elapsed time in seconds
    elapsed_time = (current_time - start_time).total_seconds()

    # Calculate estimated remaining time in seconds
    if i > 0:  # To avoid division by zero
        estimated_remaining_time = (elapsed_time / i) * (num_iterations - i - 1)
    else:
        estimated_remaining_time = elapsed_time * (num_iterations - 1)

    # Calculate estimated end time
    estimated_end_time = current_time + datetime.timedelta(seconds=estimated_remaining_time)

    # Convert times to appropriate formats
    elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
    estimated_remaining_time_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
    estimated_end_time_str = estimated_end_time.strftime("%Y/%m/%d/%H:%M:%S")

    # Adjust the estimated end time to JST timezone (UTC+9)
    # estimated_end_time = estimated_end_time + datetime.timedelta(hours=9)
    # estimated_end_time_jst_str = estimated_end_time.strftime("%Y/%m/%d/%H:%M:%S")


    print(f"Iteration: {i+1}")
    print(f"Current time: {current_time_str}")
    print(f"    Execution time so far: {elapsed_time_str}")
    print(f"    Estimated remaining time: {estimated_remaining_time_str}")
    # print(f"Estimated end time (UTC): {estimated_end_time_str}")
    print(f"Estimated end time: {estimated_end_time_str}")
    print("\n")

    
    
    
def KickPGM(start_time):
    # '2023/10/11 03:30'
    t = datetime.datetime.strptime(start_time, '%Y/%m/%d %H:%M')
    t2 = (t - datetime.datetime.now()).seconds
    time.sleep(t2)
    return 



def fun_MakeDir(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
            
            
            
def add_latex_text(x, y, text, fontsize=12, **kwargs):
    from matplotlib import rc
    rc('text', usetex=True)
    # rc('text.latex', preamble=r'\usepackage{cmbright}')
    
    plt.text(x, y, text, fontsize=fontsize, **kwargs)
    rc('text', usetex=False)


## Function for 1D scaling plot
def fun_PltScaling(
    df, xname, yname, savename,
    xlabel=None, 
    ylabel=None, 
    title=None,
    colors={
        'ci' : sns.color_palette('binary', 10)[5],
        'opt' : sns.color_palette('binary', 10)[8],
        'mean' : sns.color_palette('Blues', 25)[-1]
    }, 
    formula=False, 
    popt=None, 
    func=None, 
    xticks=None, 
    fig=None, 
    place=(1,1,1)
):
    
    ci_color = colors['ci']
    opt_color = colors['opt']
    mean_color = colors['mean']

        
    df1 = df[df[xname]>0]
    df2 = df[df[xname]==0]
    
    if not df1.empty:   
        def func(x, a, A, B):
            return A * np.power(x, -a) + B

        grouped_mean = df1[[xname, yname]].groupby(xname).mean()
        popt, pcov = curve_fit(func, grouped_mean.index, grouped_mean[yname], maxfev=10**5, bounds=([0, 0, 0], [1, np.inf, np.inf]))
    
    
    sample_list = df1.drop_duplicates(subset=xname)[xname].sort_values()
    
    fig_flg=0
    if fig is None:
        fig = plt.figure()
        fig_flg = 1
        

    ## Scaling curve
    ax = fig.add_subplot(*place)
    ax.plot(sample_list, df1.groupby(xname)[yname].mean(), marker='o', c=mean_color, markersize=5, alpha=1, lw=2)
    
    if not df2.empty:
        ax.axhline(y=df2[yname].mean(), c='grey', alpha=0.75, lw=1, linestyle='-.')
        


    ## Confidence interval
    if df1.groupby(xname)['Seed'].nunique().min() > 1:
        plt_ns = []
        bst_low = []
        bst_high = []
        for n_sample in sample_list:
            plt_ns.append(n_sample)
            bst = bootstrap((df1[df1[xname]==n_sample][yname].values,), np.mean, confidence_level=0.9, random_state=373)
            bst_low.append(bst.confidence_interval.low)
            bst_high.append(bst.confidence_interval.high)

        df_low = pd.Series(bst_low, index=plt_ns)
        df_high = pd.Series(bst_high, index=plt_ns)

        ax.fill_between(x=sample_list, 
                         y1=df_low,
                         y2=df_high, alpha=0.12, color=ci_color)


    if popt is not None:
        x_curve = np.linspace(sample_list.iloc[0], sample_list.iloc[-1], 10000)
        y_curve = func(x_curve, a=popt[0], A=popt[1], B=popt[2])
        ax.plot(x_curve, y_curve, color=opt_color, linestyle='--', alpha=0.9, linewidth=0.75, zorder=1)



    if not df1.empty: 
        ax.set_xscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    if xlabel is not None:
        ax.set_xlabel(xlabel, size=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, size=16)
    if title is not None:
        ax.set_title(title, size=20)
        
    # if xlabel is None:
    #     xlabel=xname
    # if ylabel is None:
    #     ylabel=yname
    
    if not df1.empty:  
        txts = []
        for ii in range(3):
            mantissa, exponent = f'{popt[ii]:.3e}'.split('e')
            exponent = int(exponent)
            mantissa = f'{float(mantissa):.3g}'
            if exponent>=-3:
                txts = txts + [f'{popt[ii]:,.3g}']
            else:
                txts = txts + [f'{mantissa} \\times 10^{{{exponent}}}']

        # txt = fr'${tmp_popt[1]:,.3g} N_s^{{{tmp_popt[0]:,.3g}}} + {mantissa} \times 10^{{{exponent}}}$'
        txt = fr'${txts[1]} N_s^{{-{txts[0]}}} + {txts[2]}$'

        add_latex_text(0.05, 0.075, txt, fontsize=16, ha='left', va='center', transform=plt.gca().transAxes, backgroundcolor='white')


    
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    if fig_flg==1:
        plt.savefig(savename)

        plt.show()
        
    if not df1.empty:  
        return popt
    
def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

def weighted_quantile(v, w_normalized, q):
    if (len(v) != len(w_normalized)):
        raise ValueError('Error: v is length ' + str(len(v)) + ', but w_normalized is length ' + str(len(w_normalized)))
        
    if (np.sum(w_normalized) > 1.01 or np.sum(w_normalized) < 0.99):
        raise ValueError('Error: w_normalized does not add to 1')
        
    if (q < 0 or 1 < q):
        raise ValueError('Error: Invalid q')

    n = len(v)
    
    v_sorted, w_sorted = sort_both_by_first(v, w_normalized)
    
    cum_w_sum = w_sorted[0]
    i = 0
    while(cum_w_sum <= q):
            i += 1
            cum_w_sum += w_sorted[i]
            
    if (q > 0.5): ## If taking upper quantile: ceil
        return v_sorted[i]
            
    elif (q < 0.5): ## Elif taking lower quantile:
        if (i > 0):
            return v_sorted[i-1]
        else:
            return v_sorted[0]
        
    else: ## Else taking median, return weighted average if don't have cum_w_sum == 0.5
        if (cum_w_sum == 0.5):
            return v_sorted[i]
        
        elif (i > 0):
            return (v_sorted[i]*w_sorted[i] + v_sorted[i-1]*w_sorted[i-1]) / (w_sorted[i] + w_sorted[i-1])
        
        else:
            return v_sorted[0]

def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]
    
    return [v_sorted, w_sorted]

def calc_qhat(df, alpha):
    scores = np.abs(df['Observation'].to_numpy() - df['Prediction'].to_numpy())
    threshold = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    if threshold >= len(scores):
        qhat = float('inf')
    else:
        qhat = np.sort(scores, axis=0)[threshold]
    return qhat

def calc_emp_q(df, qhat):
    emp_scores = np.abs(df['Observation'].to_numpy() - df['Prediction'].to_numpy())
    emp_Q = len(emp_scores[emp_scores <= qhat]) / len(emp_scores) 
    return emp_Q

def calc_wsplit_emp_cov(wsplit_quantiles, test_y, pred_y):
    wsplit_emp_cov = np.zeros(len(test_y))
    for i in range(len(test_y)):
        if (pred_y[i] >= test_y[i] - wsplit_quantiles[i])and(pred_y[i] <= test_y[i] + wsplit_quantiles[i]):
            wsplit_emp_cov[i] = 1
    return np.sum(wsplit_emp_cov) / len(wsplit_emp_cov)

def weight_split_CP(x, cal_idx, train_idx, test_idx, alpha, cal_y, pred_cal_y):
    resids_split = np.abs(cal_y - pred_cal_y)
    ind_split = int(np.ceil((len(resids_split) + 1) * (1 - alpha)))
    positive_infinity = np.array([float('inf')])
    unweighted_split_vals = np.concatenate([resids_split, positive_infinity])    

    weights_normalized_wsplit = np.zeros((len(cal_idx) + 1, len(test_idx)))
    wsplit_quantiles = np.zeros(len(test_idx))

    clf = LogisticRegression(random_state=0,max_iter=1000).fit(x.loc[cal_idx + train_idx + test_idx],
                                                np.r_[np.ones(len(cal_idx + train_idx)), np.zeros(len(test_idx))])
    lr_probs = clf.predict_proba(x.loc[cal_idx + train_idx + test_idx])
    weights_lr = lr_probs[:,0] / lr_probs[:,1]

    sum_cal_weights = np.sum(weights_lr[:len(resids_split)])
    for j in range(0, len(test_idx)):
        for i in range(0, len(resids_split) + 1):
            if (i < len(resids_split)):
                weights_normalized_wsplit[i, j] = weights_lr[i] / (sum_cal_weights + weights_lr[len(cal_idx + train_idx) + j])
            else:
                weights_normalized_wsplit[i, j] = weights_lr[len(cal_idx + train_idx)+j] / (sum_cal_weights + weights_lr[len(cal_idx + train_idx) + j])
        
        wsplit_quantiles[j] = weighted_quantile(unweighted_split_vals, weights_normalized_wsplit[:, j], 1 - alpha)

    return wsplit_quantiles