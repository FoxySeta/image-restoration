import main
import sys
import matplotlib
import matplotlib.pyplot as plt

decimal_places = 4

'''
    Sets matplotlib options that are shared across all our plots
'''
def plt_common():
    matplotlib.use('pgf')
    # swap pdflatex for tectonic based on your system
    plt.rc('pgf', texsystem='pdflatex',rcfonts=False)
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)

'''
    plots the three methods available to show differences side by side
'''
def plot_methods():
    # images = ['1', 'A', 'B']
    images = ['1','2','3']
    blur_factor = 2 # Selects which type of blur to use from main.blurs
    l = 0.04 # Lambda value common for all methods
    # List of methods to plot. The structur for each tuple is:
    #   (method internal name, minimize function, method display name)
    methods = (
        ('naive', 'scipy', 'Naive'),
        ('tikhonov', 'scipy', 'Tikhonov'),
        ('tikhonov', 'our', 'Tikhonov (nostro minimize)'),
        ('tv', 'scipy', 'Variazione Totale')
    )

    # Setup the plot layout
    fig, axs = plt.subplots(
        1+len(methods), len(images),
        squeeze=False, constrained_layout=True,
        figsize=(9,9)
    )
    fig.suptitle(f'Immagini corrotte e ripristinate con varie tecniche')

    blurred_images = []
    for i, image in enumerate(images):
        # Read image from file
        original = main.phase0(image)
        # Take only the needed image from the phase1 blurring
        blurred_data = main.phase1(original)[blur_factor]
        axs[0][i].set_title(f'\\texttt{{{image}.png}}\n$PSNR={round(blurred_data[1],decimal_places)}$\n$MSE={round(blurred_data[2],decimal_places)}$')
        if i == 0:
                axs[0][i].set_ylabel('Originale')
        axs[0][i].imshow(original, cmap='gray', vmin=0, vmax=1)
        blurred_images.append(blurred_data)

    for i, blurred in enumerate(blurred_images):
        for j, method in enumerate(methods):
            # Deblur each image with the appropriate metho/minimization function
            phi_dphi = (main.methods[method[0]]['phi'], main.methods[method[0]]['dphi'])
            minFun = main.sci_minimize if method[1] == 'scipy' else main.our_minimize
            deblurred = main.phasen(blurred, l, phi_dphi, minFun)
            if i == 0:
                axs[j+1][i].set_ylabel(method[2])
            axs[j+1][i].imshow(deblurred, cmap='gray', vmin=0, vmax=1)

actions = {
    'methods': plot_methods
}

if __name__ == '__main__':
    args = sys.argv[1:]
    if(len(args) == 0 or args[0] not in actions):
        print('Esempio di esecuzione:\npython plot.py [methods]')
        exit()

    action = args[0]
    plt_common()
    actions[action]()
    plt.savefig(f'report/{action}.pgf', dpi=150)
