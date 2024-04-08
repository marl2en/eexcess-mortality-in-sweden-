
"""
sudo pip3 install convertdate 
sudo pip3 install lunarcalendar
sudo pip3 install holidays

#sudo pip3 install fbprophet
sudo pip3 install prophet==1.0

sudo pip3 install Theano

sudo apt-get install hdf5-devel

"""






import csv
import datetime as dt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import numpy as np
import json

import statsmodels.tsa.api as smt
from scipy import stats

from statsmodels.tsa.api import SARIMAX
import calendar

def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return dt.date(year, month, day)

def tsplot(y, lags=None, figsize=(18, 16), style='bmh',target='Live births',show=True,takeSquare=False):
    if takeSquare: y = np.square(y); title = 'Time Series Analysis Plots of Square of'+ target
    else: title = 'Time Series Analysis Plots of '+ target
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        stats.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)
        plt.tight_layout()
    if show: plt.show()


def showDTplot(df,col2show=['realbirths','yhat','yhat_lower', 'yhat_upper'],offset=2,title='Live Births in Sweden',ylabel='Births per Month',confInt=[],target='Live births',horizon=12,year=False):
    colors = ['b','g','r','yellow','black']
    fig, axes = plt.subplots(1,1,sharex=True,figsize=(16,10))
    fig.autofmt_xdate()
    axes.set_title(title,fontsize=14)
    axes.set_ylabel(ylabel, fontsize=12)
    if year: axes.set_xlabel("Year", fontsize=12)
    else: axes.set_xlabel("Time in Month-Year", fontsize=12)
    if confInt != []:
        for i,ele in enumerate(confInt):
            conf_label = ele[0] + '-' + ele[1]
            axes.fill_between(df.index[offset:], df[ele[0]].values[offset:], df[ele[1]].values[offset:],color=colors[i], alpha=0.1)
    for ele in col2show:
        if ele == target: 
            if horizon != 0: axes.plot(df.index[offset:-horizon], df[ele][offset:-horizon],label=ele)
            else: axes.plot(df.index[offset:], df[ele][offset:],label=ele)
        else: axes.plot(df.index[offset:], df[ele][offset:],label=ele)
    if year: xfmt = mdates.DateFormatter('%Y')
    else: xfmt = mdates.DateFormatter('%m-%y')
    axes.xaxis.set_major_formatter(xfmt)
    plt.legend()
    fig.tight_layout()
    plt.show()




def saveDict(outdict,filename='/home/pi/MonitorStation/battery.json'):
    with open(filename, 'w') as outfile:
        json.dump(outdict, outfile,sort_keys=True,indent=4) # separators=(',', ': ')

def loadDict(filename='/home/pi/MonitorStation/battery.json'):
    with open(filename) as f:
        dictobj = json.load(f)
    return dictobj



def saveCSV(csvfile='',datalist=[]):
    """Save data list to csv file"""
    with open(csvfile, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in datalist:
            writer.writerow(val)

def readCSV(csvfile='', mode='rb' ):
    """Read data list from csv file"""
    data = []
    with open(csvfile, mode) as f:
        reader = csv.reader(f, delimiter=',', lineterminator='\n')
        for row in reader:
            data.append(row)
    return data

def saveCSVappend(csvfile='',datalist=[]):
    """Save data list to csv file"""
    with open(csvfile, "a") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in datalist:
            writer.writerow(val)



#numbers['Men&Women']['age']['all']['cancer_types']['all']['fitted_'+str(train_last_year )+'_diff'] = forecast['diff'].values.tolist()

# your path to documents
save_path = '...............'


################ prepare data #####################3


numbers = {}



data = readCSV(csvfile=save_path+'deaths.csv', mode='r' )


numbers['Year'] = [int(x) for x in data[0][3:] ]

numbers['Date'] = [x+'-12-31' for x in data[0][3:] ]

numbers['Men&Women'] = {'age':{},'AG5':{},'AG10':{}}
numbers['Men'] = {'age':{},'AG5':{},'AG10':{}}
numbers['Women'] = {'age':{},'AG5':{},'AG10':{}}

numobs = len(numbers['Year'])


for ele in data[1:]:
    gender = ele[2]
    age = ele[1]
    if age == '100+': age = 100
    else: age = int(age)
    ye = [int(x) for x in ele[3:]]
    print(gender,age,ye)
    numbers[gender]['age'][age] = {'deaths': ye}



saveDict(numbers,filename=save_path+'numbers.json')
numbers = loadDict(filename=save_path+'numbers.json')

#### population ####
pop = readCSV(csvfile=save_path+'population.csv', mode='r' )

pop[0][-numobs:]
['1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']



for ele in pop[1:]:
    gender = ele[1]
    age = int(ele[0])
    if age >= 100: age = 100
    else: age = int(age)
    ye = [int(x) for x in ele[-numobs:]]
    if age >= 100:
        if 'pop' in numbers[gender]['age'][str(age)].keys():
            ye = np.array(numbers[gender]['age'][str(age)]['pop']) + np.array(ye)
            ye = ye.tolist()
            numbers[gender]['age'][str(age)]['pop'] = ye
        else:
            numbers[gender]['age'][str(age)]['pop'] = ye
    else: 
        numbers[gender]['age'][str(age)]['pop'] = ye
    print(gender,age,ye)



saveDict(numbers,filename=save_path+'numbers2.json')

for age in numbers[gender]['age'].keys():
    pop = np.array(numbers['Men']['age'][age]['pop']) + np.array(numbers['Women']['age'][age]['pop'])
    pop = pop.tolist()
    deaths = np.array(numbers['Men']['age'][age]['deaths']) + np.array(numbers['Women']['age'][age]['deaths'])
    deaths = deaths.tolist()
    #print(age,deaths/pop)
    numbers['Men&Women']['age'][age] = {'deaths': deaths, 'pop': pop}
    
    



ageint = [x for x in range(0,101,10) ]

for gender in ['Men','Women','Men&Women']:
    for i,j in zip(ageint[:-1], ageint[1:] ):
        if j == 100:ag = str(i)+'_'+str(j)
        else: ag = str(i)+'_'+str(j-1)
        for age in numbers[gender]['age'].keys():
            a = int(age)
            if a == i: 
                pop = np.array(numbers[gender]['age'][age]['pop'])
                deaths = np.array(numbers[gender]['age'][age]['deaths'])
            else:
                if i < a < j:
                    pop = pop + np.array(numbers[gender]['age'][age]['pop'])
                    deaths = deaths + np.array(numbers[gender]['age'][age]['deaths'])
        print(gender,ag,pop,deaths)
        numbers[gender]['AG10'][ag] = {'pop': pop.tolist(),'deaths': deaths.tolist()}


saveDict(numbers,filename=save_path+'numbers3.json')



ageint = [x for x in range(0,101,5) ]

for gender in ['Men','Women','Men&Women']:
    for i,j in zip(ageint[:-1], ageint[1:] ):
        if j == 100:ag = str(i)+'_'+str(j)
        else: ag = str(i)+'_'+str(j-1)
        for age in numbers[gender]['age'].keys():
            a = int(age)
            if a == i: 
                pop = np.array(numbers[gender]['age'][age]['pop'])
                deaths = np.array(numbers[gender]['age'][age]['deaths'])
            else:
                if i < a < j:
                    pop = pop + np.array(numbers[gender]['age'][age]['pop'])
                    deaths = deaths + np.array(numbers[gender]['age'][age]['deaths'])
        print(gender,ag,pop,deaths)
        numbers[gender]['AG5'][ag] = {'pop': pop.tolist(),'deaths': deaths.tolist()}



########## incidence 

for gender in ['Men','Women','Men&Women']:
    for ag in numbers[gender]['AG10'].keys():
        pop = np.array(numbers[gender]['AG10'][ag]['pop'])
        deaths = np.array(numbers[gender]['AG10'][ag]['deaths'])
        i = np.round(1000*deaths/pop,4)
        print(gender,ag,i)
        numbers[gender]['AG10'][ag]['inc'] = i.tolist()



for gender in ['Men','Women','Men&Women']:
    for ag in numbers[gender]['AG5'].keys():
        pop = np.array(numbers[gender]['AG5'][ag]['pop'])
        deaths = np.array(numbers[gender]['AG5'][ag]['deaths'])
        i = np.round(1000*deaths/pop,4)
        print(gender,ag,i)
        numbers[gender]['AG5'][ag]['inc'] = i.tolist()


saveDict(numbers,filename=save_path+'numbers4.json')





def createDataset(numbers,genderlist=['Men','Women','Men&Women'],AG='AG10',val='inc',DateCol=False):
    df = pd.DataFrame() # {'Year':numbers['Year']}
    dates = [dt.date(year=x,month=12,day=31) for x in numbers['Year']]
    df.index = dates
    datesstr = [x.isoformat() for x in dates]
    if DateCol: df['Date'] = datesstr
    for gender in genderlist:
        for ag in numbers[gender][AG].keys():
            df[gender+'_'+ag] = numbers[gender][AG][ag][val]
    return df


df = createDataset(numbers,genderlist=['Men'],AG='AG10',val='inc')

df.columns # Index(['Men_0_9', 'Men_10_19', 'Men_20_29', 'Men_30_39', 'Men_40_49','Men_50_59', 'Men_60_69', 'Men_70_79', 'Men_80_89', 'Men_90_100'],  dtype='object')



showDTplot(df,col2show=['Men_0_9','Men_10_19','Men_20_29','Men_30_39','Men_40_49','Men_50_59','Men_60_69','Men_70_79','Men_80_89','Men_90_100'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)

showDTplot(df,col2show=['Men_0_9','Men_10_19','Men_20_29','Men_30_39','Men_40_49'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)

showDTplot(df,col2show=['Men_0_9','Men_10_19','Men_20_29','Men_30_39'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)
showDTplot(df,col2show=['Men_40_49','Men_50_59','Men_60_69'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)
showDTplot(df,col2show=['Men_70_79','Men_80_89','Men_90_100'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)


df = createDataset(numbers,genderlist=['Women'],AG='AG10',val='inc')

df.columns # Index(['Women_0_9','Women_10_19','Women_20_29','Women_30_39','Women_40_49','Women_50_59','Women_60_69','Women_70_79','Women_80_89','Women_90_100'], dtype='object')

showDTplot(df,col2show=['Women_0_9','Women_10_19','Women_20_29','Women_30_39'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)
showDTplot(df,col2show=['Women_40_49','Women_50_59','Women_60_69'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)
showDTplot(df,col2show=['Women_70_79','Women_80_89','Women_90_100'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)

################

numbers = loadDict(filename=save_path+'numbers4.json')

df = createDataset(numbers,genderlist=['Men'],AG='AG10',val='inc')





horizon = 4

mask = df.index < dt.datetime(2020,1,1).date()



dd = df[['Date','Men_30_39']].loc[mask].copy()

############# get order of AR ############

res_list = []
for ar in range(1,6,1):
    print('ar',ar)
    statespace_model = {'level':'smooth trend','stochastic_level':False,'trend': True,'stochastic_trend': False,'cycle': False,'irregular': True,'autoregressive':ar }
    ssm = sm.tsa.UnobservedComponents(X, **statespace_model)
    res = ssm.fit(method='powell',cov_type ='opg', disp=False) 
    fig = res.plot_components(legend_loc='lower left', figsize=(15, 9));plt.show()
    print(res.summary())
    res_list.append((ar,res.aic,res.llf))


res_list
[(1, -135.22623021403368, 71.61311510701684), 
(2, -133.6550753556171, 71.82753767780855), 
(3, -131.65084449651854, 71.82542224825927), 
(4, -129.1776834323614, 71.5888417161807), 
(5, -128.1564709946823, 72.07823549734115)]

####################################################




def prepareDataset(numbers,year2train=2019,genders=['Men'],AG='AG10',show=True,calcSSM= True):
    df = createDataset(numbers,genderlist=genders, AG=AG,val='inc')
    real_y = df.columns.tolist()
    print('dispersion',df[real_y].var()/df[real_y].mean())
    mask = df.index <= dt.date(year2train,12,31)
    steps = len(df) - mask.sum()
    print('steps',steps)
    params = []
    if calcSSM:
        statespace_model = {'level':'smooth trend','stochastic_level':False,'trend': True,'stochastic_trend': False,'cycle': False,'irregular': True,'autoregressive':1 }
        df_train1 = df[mask].copy()
        for col in real_y:
            y = df_train1[col].values.flatten()
            ssm = sm.tsa.UnobservedComponents(y, **statespace_model)
            res = ssm.fit(method='bfgs',cov_type ='robust', disp=False) # -8513.184136521999
            #f = res.get_forecast(steps=steps,alpha=0.05)
            #fitted = np.array([res.fittedvalues[1]] + res.fittedvalues[1:].tolist())
            pred = res.get_prediction(start=1,end=len(df))
            mean = pred.summary_frame()['mean'].values.tolist()
            df[col+'_pred'] = mean
            lower = pred.summary_frame()['mean_ci_lower'].values.tolist()[1:]
            lower = [lower[0]] + lower
            df[col+'_pred_lower'] = lower
            upper = pred.summary_frame()['mean_ci_upper'].values.tolist()[1:]
            upper = [upper[0]] + upper
            df[col+'_pred_upper'] = upper
            if show:
                plt.plot(df.index,df[col].values,label='real')
                plt.plot(df.index,df[col+'_pred'].values,label='pred')
                plt.plot(df.index,df[col+'_pred_lower'].values,label='lower')
                plt.plot(df.index,df[col+'_pred_upper'].values,label='upper')
                plt.legend()
                plt.show()
            df[col+'_level'] = np.nan
            df[col+'_level'].iloc[mask] = res.level['smoothed']
            df[col+'_trend'] = np.nan
            df[col+'_trend'].iloc[mask] = res.trend['smoothed']
            df[col+'_ar'] = np.nan
            df[col+'_ar'].iloc[mask] = res.autoregressive['smoothed']
            df[col+'_resid'] = np.nan
            resid = res.resid[1:]
            resid = [resid[0]] + resid.tolist()
            df[col+'_resid'].iloc[mask] = resid
            params.append(res.params)
            if show: 
                fig = res.plot_components(legend_loc='lower right', figsize=(15, 9));plt.show()
                print(res.summary())
                print('aic',res.aic,'llf',res.llf)
                res.plot_diagnostics(); plt.show()
                plt.plot(res.level['smoothed'],label='s');
                plt.plot(res.level['filtered'],label='filt')
                plt.plot(y,label='org')
                plt.legend();plt.show()
                print('abs error',np.absolute(res.forecasts_error).mean(),'aic',res.aic,'pvalues',res.pvalues)
                print('scores',ssm.score(res.params))
    df_train = df[mask].copy()
    #df_test = df[~mask].copy()
    df_test = df.iloc[-len(df_train):].copy()
    num_obs = len(df_train)
    y_train_real = df_train[real_y].values
    y_pred_real = df_test[real_y].values
    if show: showDTplot(df,col2show=real_y,offset=0,title='Mortality_'+str(genders),ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)
    return df_train,df_test,real_y,params 



df_train,df_test,real_y,params = prepareDataset(numbers,year2train=2019,genders=['Men'],AG='AG10',show=False,calcSSM= True)


df_test.columns
Index(['Men_0_9', 'Men_10_19', 'Men_20_29', 'Men_30_39', 'Men_40_49',
       'Men_50_59', 'Men_60_69', 'Men_70_79', 'Men_80_89', 'Men_90_100',
       'Men_0_9_pred', 'Men_0_9_pred_lower', 'Men_0_9_pred_upper',
       'Men_0_9_level', 'Men_0_9_trend', 'Men_0_9_ar', 'Men_0_9_resid',
       'Men_10_19_pred', 'Men_10_19_pred_lower', 'Men_10_19_pred_upper',
       'Men_10_19_level', 'Men_10_19_trend', 'Men_10_19_ar', 'Men_10_19_resid',
       'Men_20_29_pred', 'Men_20_29_pred_lower', 'Men_20_29_pred_upper',
       'Men_20_29_level', 'Men_20_29_trend', 'Men_20_29_ar', 'Men_20_29_resid',
       'Men_30_39_pred', 'Men_30_39_pred_lower', 'Men_30_39_pred_upper',
       'Men_30_39_level', 'Men_30_39_trend', 'Men_30_39_ar', 'Men_30_39_resid',
       'Men_40_49_pred', 'Men_40_49_pred_lower', 'Men_40_49_pred_upper',
       'Men_40_49_level', 'Men_40_49_trend', 'Men_40_49_ar', 'Men_40_49_resid',
       'Men_50_59_pred', 'Men_50_59_pred_lower', 'Men_50_59_pred_upper',
       'Men_50_59_level', 'Men_50_59_trend', 'Men_50_59_ar', 'Men_50_59_resid',
       'Men_60_69_pred', 'Men_60_69_pred_lower', 'Men_60_69_pred_upper',
       'Men_60_69_level', 'Men_60_69_trend', 'Men_60_69_ar', 'Men_60_69_resid',
       'Men_70_79_pred', 'Men_70_79_pred_lower', 'Men_70_79_pred_upper',
       'Men_70_79_level', 'Men_70_79_trend', 'Men_70_79_ar', 'Men_70_79_resid',
       'Men_80_89_pred', 'Men_80_89_pred_lower', 'Men_80_89_pred_upper',
       'Men_80_89_level', 'Men_80_89_trend', 'Men_80_89_ar', 'Men_80_89_resid',
       'Men_90_100_pred', 'Men_90_100_pred_lower', 'Men_90_100_pred_upper',
       'Men_90_100_level', 'Men_90_100_trend', 'Men_90_100_ar',
       'Men_90_100_resid'],
      dtype='object')

confInt = [('Men_0_9_pred_lower', 'Men_0_9_pred_upper'),('Men_10_19_pred_lower', 'Men_10_19_pred_upper'),('Men_20_29_pred_lower', 'Men_20_29_pred_upper'),('Men_30_39_pred_lower', 'Men_30_39_pred_upper')]


showDTplot(df_test,col2show=['Men_0_9','Men_10_19','Men_20_29','Men_30_39'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=confInt,target='',horizon=0,year=True)

showDTplot(df_test,col2show=['Men_0_9','Men_10_19','Men_20_29','Men_30_39','Men_0_9_pred','Men_10_19_pred','Men_20_29_pred','Men_30_39_pred'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)





showDTplot(df,col2show=['Men_40_49','Men_50_59','Men_60_69'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)
showDTplot(df,col2show=['Men_70_79','Men_80_89','Men_90_100'],offset=0,title='Deaths per Age Group',ylabel='Deaths/1000 per Year',confInt=[],target='',horizon=0,year=True)






