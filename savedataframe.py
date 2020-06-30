# Nic's data frame saving module
# # import savingdataframe as svdf
# # save_dat(average_temps_in_florida, "average_temps_in_florida", True)

import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table

IMAGES_PATH = "folder_name" # Where you want them to save to

def save_dat(data_id, file_name, state, fig_extension="png", resolution=600):
    if state: # allows you to disable saving to save a huge amount of time
    
        path = os.path.join(IMAGES_PATH, file_name + "." + fig_extension)
        ax = plt.subplot(111, frame_on=False)  # no visible frame
        plt.tight_layout() # may or may not want to turn this off
        plt.title(file_name)
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis

        table(ax, data_id, loc='center')  # where df is your data frame
        plt.savefig(path, format=fig_extension, dpi=resolution)

        print("\n", 50 * "-", "\n", "Saving figure", file_name, "\n", 50 * "-", "\n")
        plt.close()
