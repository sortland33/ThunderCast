#Author: Stephanie M. Ortland
import pandas as pd
import random
import numpy as np
import sys, os
from datetime import datetime,timedelta

#Determine statistics for the sea breeze case studies. 
if __name__ == "__main__":
    file_path = '/home/sortland/ci_ltg/sea_breeze_cases.csv'
    df = pd.read_csv(file_path, parse_dates=[2, 3, 4, 5])
    df['Dates'] = df['radarTime'].dt.date
    df['Time'] = df['radarTime'].dt.time
    all_dates = df['Dates'].unique()
    TC_LeadTimes = []
    Times_30dBZ_to_Lightning = []
    for dt in all_dates:
        df_sub = df[df['Dates'] == dt]
        rows = list(np.arange(0, len(df_sub)))
        sample = random.sample(rows, 1)
        TC_LeadTimes.append(df_sub.iloc[sample[0]]['TCtimetoRadar'])
        Times_30dBZ_to_Lightning.append(df_sub.iloc[sample[0]]['TimeDiffRadarLightning'])
    t_95_values = {'1': 12.71, '2': 4.303, '3': 3.182, '4': 2.776, '5': 2.571, '6': 2.447, '7': 2.365, '8': 2.306,
                   '9': 2.262, '10': 2.228, '11': 2.201, '12':2.179, '13':2.160, '14':2.145, '15':2.131, '16':2.120,
                   '17':2.110, '18':2.101, '19':2.093, '20':2.086, '21':2.080, '22':2.074, '23':2.069, '24':2.064,
                   '25':2.060, '26':2.056, '27':2.052, '28':2.048, '29':2.045, '30':2.042}
    n = len(TC_LeadTimes)
    ci_TC =  (np.std(TC_LeadTimes))/np.sqrt(n) * t_95_values[str(int(n-1))]#confidence interval sigma_hat = sigma/sqrt(n) -> ci = sigma_hat*t_95_values
    ci_RtoL = (np.std(Times_30dBZ_to_Lightning))/np.sqrt(n) * t_95_values[str(int(n-1))]
    print('TC Leadtime Mean, confidence interval', np.mean(TC_LeadTimes), ci_TC)
    print('Radar to Lightning Mean, Std', np.mean(Times_30dBZ_to_Lightning), ci_RtoL)