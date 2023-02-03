import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy


def latex_tabular_writer(excel, metrics_on_datasets_formal_names, source_name, target_name, dp):
    '''
    Input:
        <pandas.DataFrame>: Contains the spreadsheet of performances, where each row is a metric on an instance set, and every column contains the epoch.
        <dict>: Each key contains each spreadsheet's name, and its corresponding value is the more formal equivalent in LaTeX code.
        <str>: Architecture's name.
        <str>: Dataset's name.
        <str>: Directories path where the .tex file will be written.
    '''

    if target_name == '':
        output_rfp = '%s.tex'%(source_name)
    else:
        output_rfp = '%s_%s.tex'%(source_name, target_name)

    if target_name == '':
        initial_chunk = \
    r'''    ~\\
    {
    \centering
    \hspace*{-1.3cm}
    \begin{tabular}{ |p{1.5cm}||c|c|c|c|c|c|c|c|c|c|  }
        \hline
        \multicolumn{11}{|c|}{Training \ttt{%s}} \\
        \hline
        '''\
        %(source_name)
    else:
        initial_chunk = \
    r'''    ~\\
    {
    \centering
    \hspace*{-1.3cm}
    \begin{tabular}{ |p{1.5cm}||c|c|c|c|c|c|c|c|c|c|  }
        \hline
        \multicolumn{11}{|c|}{Training \ttt{%s\textunderscore %s}} \\
        \hline
        '''\
        %(source_name, target_name)

    ## ! Middle chunk: Begin

    middle_chunk = ''

    ## Epoch row
    for epoch in range(len(excel.keys())):
        middle_chunk = middle_chunk + r' & %d'%(epoch)
    middle_chunk = middle_chunk + \
    r''' \\
        \hline
        \hline
        '''

    for df_row_idx in excel.index:

        middle_chunk = middle_chunk + metrics_on_datasets_formal_names[df_row_idx]

        if 'loss' in df_row_idx:
            optimal_value = round(excel.loc[df_row_idx].min(), 4)
        elif 'accuracy' in df_row_idx:
            optimal_value = round(excel.loc[df_row_idx].max(), 4)

        for epoch in range(len(excel.keys())):
            rounded_value = round(excel.loc[df_row_idx][epoch], 4)
            if rounded_value == optimal_value:
                middle_chunk = middle_chunk + ' & $\optimalcellvalueformat %.4f$'%(rounded_value)
            else:
                middle_chunk = middle_chunk + ' & %.4f'%(rounded_value)
        middle_chunk = middle_chunk + \
        r''' \\
        \hline
        '''

    middle_chunk = middle_chunk[:-4]

    ## ! Middle chunk: End

    if target_name == '':
        final_chunk = \
    r'''\end{tabular}
    \captionof{table}{} \label{Table: %s, performance}
    }
    ~\\'''\
        %(source_name)
    else:
        final_chunk = \
    r'''\end{tabular}
    \captionof{table}{} \label{Table: %s_%s, performance}
    }
    ~\\'''\
        %(source_name, target_name)

    tabular_string = initial_chunk+middle_chunk+final_chunk

    with open(dp+output_rfp, 'w') as writer:
        writer.write(tabular_string)


plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.latex.preamble': r'\usepackage{amsfonts}',
})

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

experiment_rdp = './models/'

metric_names = ('loss', 'accuracy')
source_names = ('mnistsrc', 'imdbsrc')
target_names = ('', 'mnisttgt', 'imdbtgt')

metrics_on_datasets_formal_names = {'loss': r'$\hat{\mathcal{L}}_{\mathbb{D}_{\mathrm{tr}}}$', 'val_loss': r'$\mathcal{L}_{\mathbb{D}_{\mathrm{va}}}$', 'accuracy': r'$\hat{\mathrm{acc}}(\mathbb{D}_{\mathrm{tr}})$', 'val_accuracy': r'$\mathrm{acc}(\mathbb{D}_{\mathrm{va}})$'}

figure_bottom_text = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)', 4: '(e)', 5: '(f)'}

for source_name in source_names:
    ## Each line will contain 2 rows and 2 columns. Every row will be dedicated to a separate dataset, every column to a separate metric. The left columns will be the graph of loss and the right columns will be the graph of accuracy.
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(r'$\mathrm{Trainings \ Based \ on \ the \ Source \ Model \ with \ Architecture \ %s}$'%(source_name), fontsize=16)
    ax = [None for _ in range(len(target_names)*len(metric_names))]
    ## This will be used for the titles of each dataset (on each subplot row).
    row_ax = [None for _ in range(len(target_names))]
    x = np.arange(0, 10) # Epoch arrangement

    ax_idx = 0
    for target_idx in range(len(target_names)):
        target_name = target_names[target_idx]

        row_ax[target_idx] = fig.add_subplot(len(target_names), 1, target_idx+1)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        row_ax[target_idx].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        row_ax[target_idx]._frameon = False
        row_ax[target_idx].set_xticks([])
        row_ax[target_idx].set_yticks([])

        if target_name == '':
            excel_name = '%s'%(source_name)
            row_ax[target_idx].set_title(r'$\mathrm{%s}$'%(source_name), y=1.04, fontsize=11)
        else:
            excel_name = '%s_%s'%(source_name, target_name)
            row_ax[target_idx].set_title(r'$\mathrm{%s\textunderscore %s}$'%(source_name, target_name), y=1.04, fontsize=11)

        excel = pd.read_excel(io='%s%s.xlsx'%(experiment_rdp, excel_name), index_col=0)

        latex_tabular_writer(excel, metrics_on_datasets_formal_names, source_name, target_name, experiment_rdp)

        for metric_idx in range(len(metric_names)):
            metric_name = metric_names[metric_idx]

            ax[ax_idx] = fig.add_subplot(len(target_names), len(metric_names), ax_idx+1)

            ## ! Training Loss: Begin

            y = excel.loc[metric_name]
            ax[ax_idx].plot(x, y, color='blue')

            ## ! Training Loss: End

            ## ! Validation Loss: Begin

            y = excel.loc['val_'+metric_name]
            ax[ax_idx].plot(x, y, color='orange')

            ## ! Validation Loss: End

            ax[ax_idx].legend([metrics_on_datasets_formal_names[metric_name], metrics_on_datasets_formal_names['val_'+metric_name]])
            ax[ax_idx].set_xticks(range(10))
            ax[ax_idx].set_xlabel(r'epoch')
            ax[ax_idx].set_ylabel(metric_name)
            ax[ax_idx].text(0.5, -0.43, r'$\mathrm{%s}$'%(figure_bottom_text[ax_idx]), horizontalalignment='center', transform=ax[ax_idx].transAxes, fontsize=13)
            ax[ax_idx].grid()

            ax_idx = ax_idx + 1

    fig.subplots_adjust(hspace=0.64, wspace=0.19)
    # plt.show()
    # exit()
    plt.savefig('%s%s_eval_plots.png'%(experiment_rdp, source_name), dpi=300)