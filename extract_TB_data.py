#Author: Stephanie M. Ortland
#Get information out of a tensorboard file (i.e. validation stats)
import os, sys
import pandas as pd
from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as md
import matplotlib.gridspec as gridspec
from datetime import datetime,timedelta
import errno

def mkdir_p(path):
    try:
        os.makedirs(path)
    except (OSError, IOError) as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_columns(plot_type): #only grab columns of interest of each plot type
    if plot_type == 'loss':
        columns = ['step', 'train_loss_epoch', 'val_loss_epoch', 'epoch']
    elif plot_type == 'accuracy':
        columns = ['step', 'epoch', 'val_acc_epoch']
    elif plot_type == 'precision':
        columns = ['step', 'epoch', 'val_prec_epoch']
    elif plot_type =='recall':
        columns = ['step', 'epoch', 'val_recall_epoch']
    elif plot_type == 'acc_prec_rec_F1':
        columns = ['step', 'epoch', 'val_acc', 'val_prec', 'val_recall', 'val_F1score']
    elif plot_type == 'stats_plus':
        columns = ['step', 'epoch', 'val_accuracy', 'val_precision', 'val_recall', 'val_F1score', 'val_specificity']
    else:
        print('error with columns')
        sys.exit()
    return columns

def plot_loss(means, stds, ci_95, basepath):
    print('Plotting Loss')
    fig, ax = plt.subplots()
    x = [int(i) for i in means['epoch']]
    y_train = means['train_loss_epoch']
    y_val = means['val_loss_epoch']
    err_y_train = ci_95['train_loss_epoch']
    err_y_val = ci_95['val_loss_epoch']
    ax.errorbar(x, y_train, color='tab:blue', yerr=err_y_train, fmt='s-', ecolor='tab:blue', capsize=3.5, label='Training', markersize=4)
    ax.errorbar(x, y_val, color = 'tab:green', yerr = err_y_val, fmt='o-', ecolor='tab:green', capsize=3.5, label='Validation', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_xticks(np.arange(0,int(len(x)),step=2))
    ax.set_ylabel('Mean Cross-Entropy Loss')
    ax.legend()
    figure_path = basepath + 'loss.png'
    fig.savefig(figure_path, bbox_inches='tight', dpi=200)
    return

def plot_stats_plus(means, stds, ci_95, basepath):
    #Plot the stats all on one graph
    plt.style.use('tableau-colorblind10')
    x = [int(i) for i in means['epoch']]
    y_acc = means['val_accuracy']
    y_prec = means['val_precision']
    y_recall = means['val_recall']
    y_spec = means['val_specificity']
    err_y_acc = ci_95['val_accuracy']
    err_y_prec = ci_95['val_precision']
    err_y_recall = ci_95['val_recall']
    err_y_spec = ci_95['val_specificity']
    plt.errorbar(x, y_acc, yerr=err_y_acc, color='tab:green', fmt='.-', ecolor='tab:green', capsize=3.5, label='Accuracy')
    plt.errorbar(x, y_prec, yerr = err_y_prec, color='tab:blue', fmt='.-', ecolor='tab:blue', capsize=3.5, label='Precision')
    plt.errorbar(x, y_recall, yerr=err_y_recall, color='tab:orange', fmt='.-', ecolor='tab:orange', capsize=3.5, label='Recall')
    plt.errorbar(x, y_spec, yerr=err_y_spec, color='tab:red', fmt='.-', ecolor='tab:red', capsize=3.5, label='Specificity')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0,int(len(x)),step=2))
    plt.ylabel('Mean Statistical Measure (unitless)')
    plt.legend()
    figure_path = basepath + 'acc_prec_rec_spec.png'
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)
    return

def plot_stats_stacked(means, stds, ci_95, basepath, plot_limits, colors):
    #Plot the stats with individual plots stacked vertically
    ##Initial plot formatting
    fig_width_cm = 20
    fig_height_cm = 20
    inches_per_cm = 1 / 2.54  # Convert cm to inches
    fig_width = fig_width_cm * inches_per_cm  # width in inches
    fig_height = fig_height_cm * inches_per_cm  # height in inches
    fig_size = [fig_width, fig_height]

    fig = plt.figure()
    fig.set_size_inches(fig_size)

    widths = [2]
    heights = [4, 4, 4, 4]
    gs = gridspec.GridSpec(4, 1, wspace=0.1, hspace=0.3, width_ratios=widths, height_ratios=heights)
    axes = []
    num_axes = np.arange(0, 4, 1)
    plot_keys = ['val_accuracy', 'val_precision', 'val_recall', 'val_specificity']
    y_labels = ['Mean Accuracy', 'Mean Precision', 'Mean Recall', 'Mean Specificity']
    title_string = 'Validation Statistics'
    x = [int(i) for i in means['epoch']]
    for i in range(len(num_axes)):
        plot_val = plot_keys[i]
        print('plot val', plot_val)
        plot_error = ci_95[plot_val]
        plot_means = means[plot_val]
        ytick_vals = plot_limits[plot_val]
        print('tick vals', ytick_vals)
        axes.append(fig.add_subplot(gs[num_axes[i]]))
        axes[-1].errorbar(x, plot_means, yerr=plot_error, color=colors[i], fmt='.-', ecolor=colors[i], capsize=3.0)
        axes[-1].set_ylabel(y_labels[i])
        if i == (len(num_axes) - 1):
            axes[-1].set_xlabel('Epoch', labelpad=10)
        else:
            axes[-1].xaxis.set_ticklabels([])
            if i == 0:
                axes[-1].set_title(title_string, pad=10)
                axes[-1].title.set_fontsize(14)
        plt.yticks(ytick_vals)
        plt.grid()
    figure_path = basepath + 'acc_prec_rec_spec_stacked.png'
    plt.savefig(figure_path, bbox_inches='tight', dpi=200)
    return

if __name__ == "__main__":
    #Specify plot desired from the stats, trial name, and versions
    plot_type = 'stats_plus'  # options: loss, accuracy, precision, recall, HSS_TSS
    trial = 'TC_CHs_02_05_13_15'
    versions = ['version_35466137', 'version_35534054', ['version_35588758', 'version_35651174'], 'version_35659248', 'version_35699482', 'version_35768627']

    columns = get_columns(plot_type)
    plot_limits = {'val_accuracy':np.arange(.7, 1.05, .1), 'val_precision':np.arange(0.1, 0.35, 0.1), 'val_recall':np.arange(.7, 1.05, .1), 'val_specificity':np.arange(.7, 1.05, .1)}
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']

    #If more than 11 versions specified then need to add to the t_95 values using a t table
    t_95_values = {'1':12.71, '2':4.303, '3':3.182, '4':2.776, '5':2.571, '6':2.447, '7':2.365, '8':2.306, '9':2.262, '10':2.228, '11':2.201}

    #Gather stats
    dfs = {}
    max_epoch_length = 0
    trial_names = []
    for i in range(len(versions)):
        name = 'Trial'+str(i)
        trial_names.append(name)
    for i in range(len(versions)):
        basepath = '/ships19/grain/convective_init/models/' + trial + '/'
        if type(versions[i]) == list:
            dfs_temp = []
            for j in range(len(versions[i])):
                log_dir = basepath + 'lightning_logs/' + versions[i][j]
                reader = SummaryReader(log_dir, pivot=True)
                df = reader.scalars
                if i == 0:
                    for col in df.columns:
                        print(col)
                df['epoch'] = df['epoch'].str[0]
                df = df[columns]
                df.dropna(inplace=True)
                new_index = list(range(len(df['epoch'])))
                df.reindex(new_index)
                df = df.astype({'step': 'int64', 'epoch': 'int64'})
                epoch_length = len(df['epoch'])
                if epoch_length > max_epoch_length:
                    max_epoch_length = epoch_length
                df.reset_index(inplace=True, drop=True)
                dfs_temp.append(df)
            df = pd.concat(dfs_temp)
        else:
            log_dir = basepath + 'lightning_logs/' + versions[i]
            reader = SummaryReader(log_dir, pivot=True)
            df = reader.scalars
            if i == 0:
                for col in df.columns:
                    print(col)
            df['epoch'] = df['epoch'].str[0]
            df = df[columns]
            df.dropna(inplace=True)
            new_index = list(range(len(df['epoch'])))
            df.reindex(new_index)
            df = df.astype({'step': 'int64', 'epoch': 'int64'})
            epoch_length = len(df['epoch'])
            if epoch_length > max_epoch_length:
                max_epoch_length = epoch_length
            df.reset_index(inplace=True, drop=True)
        dfs[trial_names[i]] = df

    print('Trial Names', trial_names)
    df = pd.concat(dfs.values(), axis=0, keys=trial_names)
    print(df.head())
    means = {}
    stds = {}
    ci_95 = {}
    ns = {}

    for j in range(max_epoch_length):
        tmp = df[df['epoch'] == j]
        n = len(tmp)
        ns[j] = n
        t_val = t_95_values[str(n)]
        m = tmp.mean(axis=0, skipna=True)
        s = tmp.std(axis=0, skipna=True)
        for col in columns:
            if j == 0:
                means[col] = [m[col]]
                stds[col] = [s[col]/np.sqrt(n)]
                ci = stds[col]*t_val
                ci_95[col] = [ci]
            else:
                new_m = means[col]
                new_s = stds[col]
                new_ci = ci_95[col]
                new_m.append(m[col])
                new_s.append(s[col]/np.sqrt(n))
                new_ci.append(stds[col]*t_val)
                means[col] = new_m
                stds[col] = new_s
                ci_95[col] = new_ci


    basepath = '/ships19/grain/convective_init/stats/TC_CHs_02_05_13_15/'
    mkdir_p(basepath)
    #plot the stats
    if plot_type=='loss':
        plot_loss(means, stds, ci_95, basepath)
    if plot_type == 'stats_plus':
        plot_stats_stacked(means, stds, ci_95, basepath, plot_limits, colors)
        #plot_stats_plus(means, stds, ci_95, basepath)

    print('plotting complete')