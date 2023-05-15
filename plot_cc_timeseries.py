import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import sys, os
import matplotlib.dates as md
import matplotlib.gridspec as gridspec

df = pd.DataFrame(columns=['region', 'timestamp', 'prediction', 'radar', 'pt_lon', 'pt_lat'])
if __name__ == "__main__":
    date = '2022-07-06'
    lightning_obs = {'KSC-IA': datetime(2022, 7, 6, 17, 31)}
    radar_obs = {'KSC-IA': datetime(2022, 7, 6, 18, 1)}
    twenty_perc = {'KSC-IA': datetime(2022, 7, 6, 15, 41, 18)}
    #twenty_perc = {}
    root = '/ships19/grain/convective_init/CCFL/2022-07-04/'
    file_path = root + 'leadtime_3.txt'
    convect_dbz = 30
    plot_limits = {'prediction':np.arange(.1, 1, .2), 'radar':np.arange(0, 60, 10)}

    df = pd.read_fwf(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
    #df['time'] = df['timestamp'].dt.strftime('%H:%M')
    print(df.dtypes)
    print(df.head())
    #print(df.tail())
    print(df['region'].unique())
    colors = ['tab:red', 'tab:orange']
    labels = ['ThunderCast\nPrediction', 'Max. Reflectivity at\n'r'-10$^\circ$C (dB$Z$) in 7km']
    for r in df['region'].unique():
        print(r)
        if '/' in str(r):
            parts = r.split('/')
            for k in range(len(parts)):
                if k == 0:
                    reg = parts[k]
                else:
                    reg = reg + '_' + parts[k]
        else:
            reg = str(r)
        save_path = root + reg + '_timeseries.png'
        title_string = date + ' at ' + r
        df_sub = df[df['region']==r]
        plot_vals = df_sub.keys()[2:]
        print('plot_vals', plot_vals)
        lightning_time = lightning_obs[r]
        convection_time = radar_obs[r]
        prediction_time = twenty_perc[r]
        leadtime_to_convection = convection_time - prediction_time
        leadtime_to_lightning = lightning_time - prediction_time
        print('Region', r)
        print('Time of 30 dBZ:', convection_time)
        print('Leadtime to Convection:', leadtime_to_convection)
        print('Leadtime to Lightning:', leadtime_to_lightning)

        ##Initial plot formatting
        fig_width_cm = 20
        fig_height_cm = 25
        inches_per_cm = 1 / 2.54  # Convert cm to inches
        fig_width = fig_width_cm * inches_per_cm  # width in inches
        fig_height = fig_height_cm * inches_per_cm  # height in inches
        fig_size = [fig_width, fig_height]

        fig = plt.figure()
        fig.set_size_inches(fig_size)

        widths = [2]
        heights = [2, 2]
        gs = gridspec.GridSpec(2, 1, wspace=0.1, hspace=0.3, width_ratios=widths, height_ratios=heights)
        axes = []
        num_axes = np.arange(0, 2, 1)

        xlocator = md.MinuteLocator(interval=15)
        xlocator_min = md.MinuteLocator(interval=5)

        for i in range(len(num_axes)):
            plot_val = plot_vals[i]
            ytick_vals = plot_limits[plot_val]
            axes.append(fig.add_subplot(gs[num_axes[i]]))
            axes[-1].xaxis.set_major_locator(xlocator)
            axes[-1].xaxis.set_minor_locator(xlocator_min)
            axes[-1].plot(df_sub['timestamp'], df_sub[plot_vals[i]], color=colors[i])
            axes[-1].set_ylabel(labels[i])
            if i == (len(num_axes) - 1):
                axes[-1].xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
                plt.setp(axes[-1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
                axes[-1].set_xlabel('Time (Hours:Minutes UTC)', labelpad=10)
            else:
                axes[-1].xaxis.set_ticklabels([])
                if i == 0:
                    axes[-1].set_title(title_string, pad=10)
                    axes[-1].title.set_fontsize(14)
                # if i == 1:
                #     plt.axhline(convect_dbz, color='tab:brown', ls='--')
            plt.xlim(datetime(2022, 7, 4, 15, 30), datetime(2022, 7, 4, 18, 30))
            plt.axvline(lightning_time, color='tab:olive', ls='--')
            plt.axvline(convection_time, color='tab:brown', ls='--')
            plt.axvline(prediction_time, color='tab:blue', ls='--')
            plt.yticks(ytick_vals)
            plt.grid()

        fig.savefig(save_path, bbox_inches='tight', dpi=200)