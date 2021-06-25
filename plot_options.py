import matplotlib as mpl

palette = {
    "blue": "#0066ff",
    "green": "#00aa44",
    "red": "#ff2a2a"
}

def set_color_palette():
    colors = [palette["blue"], palette["green"], palette["red"]]
    mpl.rc("axes", prop_cycle=mpl.cycler(color=colors))