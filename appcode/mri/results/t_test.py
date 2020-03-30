import numpy as np
import scipy.stats as stats
import os
DIR = "/HOME/predict/20_percent_all"
metrics = ['PSNR']
for metric in metrics:
    proposed_file = os.path.join(DIR,metric+'_predict.dat')

    competitive_file = os.path.join(DIR,metric+'_RecPF.dat')
    proposed = np.fromfile(proposed_file)
    competitive = np.fromfile(competitive_file)
    print(stats.ttest_ind(proposed[1:1000],competitive[1:1000]))
    print(stats.ttest_ind_from_stats(np.mean(proposed),np.std(proposed),len(proposed),np.mean(competitive),np.std(competitive),len(competitive),False))