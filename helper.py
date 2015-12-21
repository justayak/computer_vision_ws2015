import matplotlib.pyplot as plt

def draw_mats(mats):
    cols = 2
    rows = (len(mats)- 1) // cols + 1 
   
    print("rows " + str(rows))
    return

    fig, plots = plt.subplots(
        ncols=cols,
        nrows=rows,
        sharex=True,
        sharey=True,
        subplot_kw={'adjustable':'box-forced'}

    for mat in mats:
        pass
