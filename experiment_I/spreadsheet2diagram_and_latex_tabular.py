import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy


def latex_tabular_writer(excel, metrics_on_datasets_formal_names, python_version, architecture_name, dataset_name):
    '''
    Input:
        <pandas.DataFrame>: Contains the spreadsheet of performances, where each row is a metric on an instance set, and every column contains the epoch.
        <dict>: Each key contains each spreadsheet's name, and its corresponding value is the more formal equivalent in LaTeX code.
        <int>: Python's version.
        <str>: Architecture's name.
        <str>: Dataset's name.
    '''

    output_rfp = 'python%d/models/arch_opt4_%s_dataset_%s.tex'%(python_version, architecture_name, dataset_name)

    if dataset_name == 'imdb':
        precise_dataset_name = dataset_name[:-1].upper() + dataset_name[-1]
    else:
        precise_dataset_name = deepcopy(dataset_name).upper()

    initial_chunk = \
    r'''    ~\\
    {
    \centering
    \hspace*{-1.3cm}
    \begin{tabular}{ |p{1.5cm}||c|c|c|c|c|c|c|c|c|c|  }
        \hline
        \multicolumn{11}{|c|}{Training \ttt{arch\textunderscore opt4%s\textunderscore dataset\textunderscore %s} on %s using \texttt{Python%d}} \\
        \hline
        '''\
    %(architecture_name, dataset_name, precise_dataset_name, python_version)

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

    final_chunk = \
    r'''\end{tabular}
    \captionof{table}{} \label{Table: arch_opt4%s_dataset_%s, python%d, performance}
    }
    ~\\'''\
    %(architecture_name, dataset_name, python_version)

    tabular_string = initial_chunk+middle_chunk+final_chunk

    with open(output_rfp, 'w') as writer:
        writer.write(tabular_string)


plt.rcParams.update({
    'font.size': 8,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}',
    'text.latex.preamble': r'\usepackage{amsfonts}',
})

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

metric_names = ('loss', 'accuracy')
python_versions = (2, 3)
architecture_names = ('mnist', 'imdb')
dataset_names = ('mnist', 'imdb')

metrics_on_datasets_formal_names = {'loss': r'$\hat{\mathcal{L}}_{\mathbb{D}_{\mathrm{tr}}}$', 'val_loss': r'$\mathcal{L}_{\mathbb{D}_{\mathrm{va}}}$', 'accuracy': r'$\hat{\mathrm{acc}}(\mathbb{D}_{\mathrm{tr}})$', 'val_accuracy': r'$\mathrm{acc}(\mathbb{D}_{\mathrm{va}})$'}

figure_bottom_text = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'}

for python_version in python_versions:
    for architecture_name in architecture_names:
        ## Each line will contain 2 rows and 2 columns. Every row will be dedicated to a separate dataset, every column to a separate metric. The left columns will be the graph of loss and the right columns will be the graph of accuracy.
        fig = plt.figure(figsize=(9, 7))
        if architecture_name == 'imdb':
            precise_architecture_name = architecture_name[:-1].upper() + architecture_name[-1]
        else:
            precise_architecture_name = deepcopy(architecture_name).upper()
        fig.suptitle(r'\texttt{Python%d}, $\mathrm{Trainings \ Based \ on \ the \ Architecture \ Tuned \ for \ %s}$'%(python_version, precise_architecture_name), fontsize=16)
        ax = [None for _ in range(2*len(metric_names))]
        ## This will be used for the titles of each dataset (on each subplot row).
        row_ax = [None for _ in range(len(dataset_names))]
        x = np.arange(0, 10) # Epoch arrangement

        ax_idx = 0
        for dataset_idx in range(len(dataset_names)):
            dataset_name = dataset_names[dataset_idx]

            row_ax[dataset_idx] = fig.add_subplot(len(dataset_names), 1, dataset_idx+1)
            if dataset_name == 'imdb':
                precise_dataset_name = dataset_name[:-1].upper() + dataset_name[-1]
            else:
                precise_dataset_name = deepcopy(dataset_name).upper()

            row_ax[dataset_idx].set_title(r'$\mathrm{%s \ Dataset}$'%(precise_dataset_name), y=1.04, fontsize=11)
            # Turn off axis lines and ticks of the big subplot 
            # obs alpha is 0 in RGBA string!
            row_ax[dataset_idx].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
            # removes the white frame
            row_ax[dataset_idx]._frameon = False
            row_ax[dataset_idx].set_xticks([])
            row_ax[dataset_idx].set_yticks([])

            excel_name = 'arch_opt4_%s_dataset_%s'%(architecture_name, dataset_name)

            excel = pd.read_excel(io='./python%d/models/%s.xlsx'%(python_version, excel_name), index_col=0)

            latex_tabular_writer(excel, metrics_on_datasets_formal_names, python_version, architecture_name, dataset_name)

            for metric_idx in range(len(metric_names)):
                metric_name = metric_names[metric_idx]

                ax[ax_idx] = fig.add_subplot(len(dataset_names), len(metric_names), ax_idx+1)

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
                ax[ax_idx].text(0.5, -0.27, r'$\mathrm{%s}$'%(figure_bottom_text[ax_idx]), horizontalalignment='center', transform=ax[ax_idx].transAxes, fontsize=13)
                ax[ax_idx].grid()

                ax_idx = ax_idx + 1

        fig.subplots_adjust(hspace=0.43, wspace=0.19)
        # plt.show()
        # exit()
        plt.savefig('python%d/models/python%d_%s_eval_plots.png'%(python_version, python_version, architecture_name), dpi=300)