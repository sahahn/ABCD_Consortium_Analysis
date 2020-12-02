import os
from info import contrasts

for task in contrasts:
    for contrast in range(len(contrasts[task])):
        for i in [0, 1]:

            cmds = task + ' ' + str(contrast) + ' ' + str(i)
            os.system('sbatch generate.sh ' + cmds)
            os.system('sbatch run_rely.sh ' + cmds)
            os.system('sbatch run_corr_rely.sh ' + cmds)