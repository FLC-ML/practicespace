# Nic's Figure Saving Module
# # import savingfigures as svfg
# # save_fig("name_of_figure", True)
import os
import matplotlib.pyplot as plt

# Where to save the figures
IMAGES_PATH = "images"

def save_fig(fig_id,state, tight_layout=True, fig_extension="png", resolution=300):
    if state:
        path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
        print("\n", 50*"-","\n","Saving figure", fig_id,"\n", 50*"-","\n")
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
        plt.close()