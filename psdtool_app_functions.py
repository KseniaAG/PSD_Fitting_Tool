import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy import special
from scipy.stats import geninvgauss
from matplotlib.gridspec import GridSpec


### ====  Step: Check the sample representativeness
def check_samples(y, D):
    dupl = y[y.duplicated(keep = False)].value_counts()

    if np.max(y) > 100: 
        print('⚠️ ERROR: The sample PSD has cumulative frequency over 100%! \nThe sample PSD fitting cannot be carried on')

    elif all(i<= j for i, j in zip(y, y[1:])) == False:
        print('⚠️ ERROR: The sample PSD has decreasing cumulative frequency with increasing particle sizes! \nThe sample PSD fitting cannot be carried on')

    elif all(i<= j for i, j in zip(D, D[1:])) == False:
        print('⚠️ ERROR: The sample PSD has decreasing particle sizes! \nThe sample PSD fitting cannot be carried on')

    elif (y[0] == y[-1]):
        print('⚠️ ERROR: The sample PSD is a horizontal line! \nThe sample PSD fitting cannot be carried on')

    elif (y[0] == y[-2]) & (y[0] != y[-1]):
        print('⚠️ ERROR: 90% of the sample mass in the one largest grain size grades! \nThe sample PSD fitting cannot be carried on')

    elif (y[0] == y[-3]) & (y[0] != y[-2]) & (y[0] != y[-1]):
        print('⚠️ ERROR: 90% of the sample mass in the two largest grain size grades! \nThe sample PSD fitting cannot be carried on')

    elif np.max(y) < 85:
        print('⚠️ WARNING: The sample PSD maximum percentile is less than 85%! The sample PSD fitting could be biased')

    elif np.min(y) > 15:
        print('⚠️ WARNING: The sample PSD minimum percentile is more than 15%! The sample PSD fitting could be biased')

    elif dupl.max() > 2 and (y.min() + 15) <= dupl.idxmax() <= (y.max()-15):
        print('⚠️ WARNING: The sample PSD has more than 2 duplicated percentiles, which is sign of bimodality! \nThe sample PSD fitting cannot be carried on')

    else: 
        print('✅ Success! The sample PSD satisfies all specified sorting conditions!')


### ====  Step: Sample statistics
## function to calculate representative D value from the sample via linear interpolation
def Dvalue_la(D, log2_D, y, DV):
    ## define the measured percentiles smaller and larger than searching % 
    a = np.where(y <= DV)[0]
    b = np.where(y >= DV)[0]

    ## define sieve size smaller and larger than D value
    if len(a) == 0:
        dsmaller = 0
    else: 
        dsmaller = y[a[len(a)-1]]
    
    if len(b) == 0:
        dlarger = 100
    else:
        dlarger = y[b[0]]

    ## define #log2(D) : sieve size smaller and larger than D value
    if len(a) == 0:
        Ssmaller = -8
    else: 
        Ssmaller = log2_D[a[len(a)-1]]

    if len(b) == 0:
        Slarger = 8
    else: 
        Slarger = log2_D[b[0]]

    ## define the D value
    if (Ssmaller >= Slarger):
        Dvalue = 2**Ssmaller
    elif (dsmaller==0) & (dlarger==100):
        Dvalue = np.nan#2**Ssmaller         
    else:
        Dvalue= 2**(((DV-dsmaller)/((dlarger-dsmaller)/(Slarger-Ssmaller)))+Ssmaller)

    if (len(a) > 1) & (len(b) > 1): 
        if (a[-1] == b[1]): 
            print(f'same % values in sample:, {y[a[-1]]}, {y[b[1]]}')
            Dvalue = 2**Ssmaller

    return Dvalue

## function to calculate sample statistics by formulas suggested by Folk and Ward (1957)
def check_stat(y, D):
    log2_D = np.log2(D)
    d05 = Dvalue_la(D, log2_D, y, 5)
    d16 = Dvalue_la(D, log2_D, y, 16)
    d25 = Dvalue_la(D, log2_D, y, 25)
    d50 = Dvalue_la(D, log2_D, y, 50)
    d75 = Dvalue_la(D, log2_D, y, 75)
    d84 = Dvalue_la(D, log2_D, y, 84)
    d95 = Dvalue_la(D, log2_D, y, 95)

    phi05, phi16, phi25, phi50, phi75, phi84, phi95 = -np.log2(d05), -np.log2(d16), -np.log2(d25), -np.log2(d50), -np.log2(d75), -np.log2(d84), -np.log2(d95)

    mean_size = (phi16 + phi50 + phi84) / 3
    stand_div = (phi84 - phi16) / 4 + (phi95 - phi05) / 6.6
    skew = (phi84 + phi16 - 2*phi50) / (2*(phi84 - phi16)) + (phi05 + phi95 - 2*phi50) / (2*(phi95 - phi05))
    kurtosis = (phi95 - phi05) / (2.44 * (phi75 - phi25))

    dict = {'D50[mm]' : d50,
            'Mean' : mean_size, 
            'STD': stand_div, 
            'Skewness': skew, 
            'Kurtosis': kurtosis}
    
    print('The sample PSD statistics:')
    for k in dict.keys():
        print(k + ': ', dict[k])

    return dict

### ====  Step: Fitting Functions
## Lognormal: parametr A - is sigma and parameter B - is mu (wiki page)
def LOGNORMAL(x, A, B):
    f = 1/2*(1+special.erf((np.log(x)-B)/(A*np.sqrt(2))))*100
    return f

## Two parameter Generalized Inverse Gaussian: parameter A - is lambda in theoretical form or 'p' in scipy form, parameter B is beta
def GIG(x, A, B):
    f = geninvgauss.cdf(x, A, B)*100
    return f

## Two paarmeter Weibull: parametr A - is lambda and parameter B - is k (wiki page)
def WEIBULL(x, A, B):
    f = (1 - np.exp(-((x/A)**B)))*100 
    return f

## function to fit the sample 
def fitting_function(function, y, D):

    try:
        try:
            parameters, covariance = curve_fit(function, D, y, maxfev = 1000000)
        except:
            print("Function need different initial guess of parameters (0.5, 2.5)")
            parameters, covariance = curve_fit(function, D, y, p0=(0.5,2.5), maxfev = 1000000)
        #pass
    
        fit_A = parameters[0]
        fit_B = parameters[1]
        
        fit_y = function(D, fit_A, fit_B)
    
        rmse = np.sqrt(mean_squared_error(y, fit_y))
        r_square = r2_score(y, fit_y)    
        p = 2
        n = len(D)
        aic_2 = n*np.log(mean_squared_error(y, fit_y)) + 2*p
        bic = n*np.log(mean_squared_error(y, fit_y)) + p*np.log(n)
    
    except: 
        print("Function cannot be fitted to the submitted sample PSD")
        rmse, r_square, aic_2, bic = np.nan, np.nan, np.nan, np.nan
        fit_A, fit_B = np.nan, np.nan
    
    print(f'The fitting function results: ')
    dict = {'BIC': bic, 
            'fitted_A': fit_A, 
            'fitted_B': fit_B, 
            'RMSE': rmse, 
            'R^2' : r_square, 
            'AIC': aic_2, 
            }
    
    for k in dict.keys():
        if k.startswith('fitted'):
            print(k.split('_')[0] + ' parameter ' + k.split('_')[1] + ': ', dict[k])
        else:
            print(k + ': ', dict[k])
                    
    return dict

### ====  Step: Fitting Functions
## function to assess the minumum for the following estimation of representative D value
def get_Dmin(min, max, DV, Function, A, B):
    x = np.linspace(min, max, 2000)
    f = 0
    i = 0
    try:
        while (f < DV-1):
            D = x[i]
            f = Function(D, A, B)
            i += 1
    except:
        D, f = np.nan, np.nan
    return D, f 

## function to estimate the representative D value
def reprep_Dvalue(y, D, DV, Function, A, B, numpoints):
    
    a = np.where(y < DV)[0]
    if len(a) > 3:
        minimum = D[a[-2]-1]
    else:
        minimum = 0

    if DV == 16:
        maximum = 100
    elif DV == 50:
        maximum = 256
    else:
        maximum = 10000
    
    Dmin, fmin = get_Dmin(minimum, maximum, DV, Function, A, B)
    if fmin > DV:
        maximum = Dmin
        numpoints = 100000
        Dmin, fmin = get_Dmin(minimum, maximum, DV, Function, A, B)
    Dmax = maximum

    x = np.linspace(Dmin, Dmax, numpoints)
    f = 0
    i = 0
    try:
        while (f < DV):
            D = x[i]
            f = Function(D, A, B)
            i += 1
    except:
        D, f = np.nan, np.nan
    
    if f > DV + 0.01:
        numpoints = 1000000
        Dmin = 0
        print('min - max')
        print(Dmin, Dmax)
        x = np.linspace(Dmin, Dmax, numpoints)
        f = 0
        i = 0
        try:
            while (f < DV):
                D = x[i]
                f = Function(D, A, B)
                i += 1
        except:
            D, f = np.nan, np.nan

    return D, f
        
### ====  Step: Ploting samples 
## function to setup the axes of the follwowing plot
def gridsetup(ax):
    ax.grid(which='major', axis='x', color='#DAD8D7', alpha=0.5, zorder = 0)
    ax.grid(which='major', axis='y', color='#DAD8D7', alpha=0.5, zorder = 0)
    ax.minorticks_off()
    ax.tick_params(direction='out')
    ax.tick_params(which ='major', axis = 'x', zorder = 0, left = True, bottom = True, right = False, top = False)
    ax.tick_params(which ='major', axis = 'y', zorder = 0, left = True, bottom = True, right = False, top = False)
    #ax.grid(which='minor', axis='x', visible=None)
    #ax.grid(which='minor', axis='y', visible=None) #linestyle=':', color='#DAD8D7', alpha=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for tickx, ticky in zip(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
        tickx.tick2line.set_visible(False)
        ticky.tick2line.set_visible(False)
    for ticx, ticy in zip(ax.xaxis.get_minor_ticks(), ax.yaxis.get_minor_ticks()):
        ticx.tick2line.set_visible(False)
        ticy.tick2line.set_visible(False)

## function to plot the sample PSD and fitted functions  
def plot_sample(y, D, func_results, save_as = False):
    plt.rcParams.update({'font.size': 14, 'font.weight':'normal', 'font.family': 'sans-serif'})
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 14
    plt.rcParams['figure.autolayout'] = True
    mpl.rcParams['pdf.fonttype']=42

    fig = plt.figure(figsize=(12, 5))

    gs = GridSpec(1,2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])

    colors = ['#008002', '#4169E2', '#FE0100']
        
## plot CDF
    ax = ax1
    gridsetup(ax)

    ## functions
    x = np.linspace(np.min(D)-2, np.max(D)+2, num=10000)
    fitted_ys = {}
    for keys in func_results.keys():
        if keys == 'Lognormal': 
            function = LOGNORMAL
        if keys == 'Weibull':
            function = WEIBULL
        if keys == 'GIG':
            function = GIG
        fitted_ys[keys] = function(x, func_results[keys]['fitted_A'], func_results[keys]['fitted_B'])
    ## ploting the fitted functions 
    for keys in fitted_ys.keys():
        if keys == 'Lognormal':
            ax.plot(x, fitted_ys[keys], color = colors[0], lw = 2, label = 'Lognormal BIC:' + str(np.round(func_results[keys]['BIC'], 2)), zorder = 2)
        
        elif keys == 'Weibull':
            ax.plot(x, fitted_ys[keys], color = colors[1], lw = 2, label = f'Weibull BIC:' + str(np.round(func_results[keys]['BIC'], 2)), zorder = 2)
        
        else:
            ax.plot(x, fitted_ys[keys], color = colors[2], lw = 2, label = f'GIG BIC:' + str(np.round(func_results[keys]['BIC'], 2)), zorder = 3)

    ax.plot(D, y.to_numpy(), color='black', linestyle='dashed', lw = 1, marker = 'o', markersize = 5, label = 'data', zorder = 3)

    ax.set_title(f'a) Cumulative Distribution plot', loc='left')
    ax.set_ylabel('Cumulative curve (%)')
    ax.set_xlabel('D (mm), log_scale')
    ax.set_xscale('log', base=2)

    ax.set_xlim((0.03, np.max(D)))
    x_ticks = ax.get_xticks()
    ax.set_xticks([0.0625] + [i for i in x_ticks if i >0.0625])
    ax.set_xticklabels([0.0625] + [i for i in x_ticks if i > 0.0625])
    ax.set_ylim((0, 100.5))
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.legend(prop = {'size': 10})

## plot the PD of the sample and fitted function
    ax = ax2
    gridsetup(ax)

    ## estimate the PD of the sample
    dy = [0*i for i in range(0, len(y))]
    for i in range(0, len(y)):
        if i == 0: 
            dy[i] = y[i]
        else: 
            dy[i] = y[i]-y[i-1]

    if all([np.log2(i).is_integer() for i in D]): 
        bar_w = -np.array(D)*0.5
    else:
        bar_w = -np.array(D)*0.2

    ax.bar(D, dy, align ='edge', width = bar_w, linestyle='solid', 
        lw = 1, fill = True, color = 'lightgrey', edgecolor = 'black', zorder = 2, label = 'data')

    ## estimate the derivative of the fitted functions
    dx = np.array([(x[i+1] + x[i])/2 for i in range(len(x)-1)])

    maximum_y = -100
    print('maximum_y: ', maximum_y)
    for keys in fitted_ys.keys():
        
        fit_dydx = np.diff(fitted_ys[keys])/np.diff(np.log2(x))

        if maximum_y < np.nanmax(fit_dydx):
            maximum_y = np.nanmax(fit_dydx)
            print('maximum_y: ', maximum_y)

        if keys == 'Lognormal':
            ax.plot(dx, fit_dydx, color = colors[0], lw = 2, label = 'Lognormal', zorder = 2)

        elif keys == 'Weibull':
            ax.plot(dx, fit_dydx, color = colors[1], lw = 2, label = 'Weibull', zorder = 2)
        
        else:
            ax.plot(dx, fit_dydx, color = colors[2], lw = 2, label = 'GIG', zorder = 3)


    ax.set_xscale('log', base=2)
    ax.set_xlim((0.03, np.max(D)))
    x_ticks = ax.get_xticks()
    ax.set_xticks([0.0625] + [i for i in x_ticks if i >0.0625])
    ax.set_xticklabels([0.0625] + [i for i in x_ticks if i > 0.0625])

    ax.set_ylim((0, max(np.max(dy), maximum_y)+5))
    ax.set_yticks([i for i in ax.get_yticks()])


    ax.set_title(f'b) Probability Density plot', loc='left')#, fontsize = 6.5)
    ax.set_xlabel('D (mm), log_scale')
    ax.set_ylabel('Probability density (%)')

# Legend of functions    
    # h, l = ax.get_legend_handles_labels()
    # legend_lines = ax.legend(h, l, ncol = 1, bbox_to_anchor=(0.98, 0.98), prop = {'size': 14, 'family': 'sans-serif'}, handlelength = 2)#
    # ax.add_artist(legend_lines)
    
    plt.tight_layout()
    plt.gcf().set_dpi(450) 
    
    if save_as == True:
        plt.savefig(save_as + '.png')
        print(f'figure_saved in "{save_as}.pdf"')
    #plt.close()
    return plt.gcf()
