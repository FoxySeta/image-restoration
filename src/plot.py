import main
import sys
import matplotlib
import matplotlib.pyplot as plt

# Number of decimal places to round for plotting
decimal_places = 3
# List of methods to plot. The structur for each tuple is:
#   (method internal name, minimize function, method display name)
methods = (
    ('naive', 'scipy', 'Naive'),
    ('tikhonov', 'scipy', 'Tikhonov'),
    ('tikhonov', 'our', 'Tikhonov (nostro minimize)'),
    ('tv', 'scipy', 'Variazione Totale')
)

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

    # Setup the plot layout
    fig, axs = plt.subplots(
        1+len(methods), len(images),
        squeeze=False, constrained_layout=True,
        figsize=(9,9)
    )
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
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
    plt.savefig('report/methods.pgf')

'''
    plots the differences in PSNR and MSE as blur, noise and regulation change
'''
def plot_vars():
    image = '1'
    img = main.phase0(image)
    defaults = {
        'blur': (9,1.3),
        'noise': 0.05,
        'lambda': 0.04,
        'iterate_over': None,
        'label': None,
    }
    variations = (
        {
            'blur': ((5,0.5),(7,1),(9,1.3),(11,1.5),(13,1.6),(15,1.7)),
            'iterate_over': 'blur',
            'label': lambda b: f'$\\sigma = {b[1]}, {b[0]}\\times {b[0]}$'
        },
        {
            'noise': (0.01,0.03,0.05,0.07,0.10,0.15,0.25),
            'iterate_over': 'noise',
            'label': lambda n: f'$z = {round(n, decimal_places)}$'
        },
        {
            'lambda': (0.02,0.04,0.08,0.16,0.32,0.64),
            'iterate_over': 'lambda',
            'label': lambda l: f'$\\lambda = {round(l, decimal_places)}$'
        }
    )
    # Plot result differences as defaults change
    for var in variations:
        table = []
        for _ in enumerate(methods):
            table.append([])
        # Make a copy of the default arguments to avoid overwriting
        params = defaults.copy()
        params.update(var)
        columns = params[params['iterate_over']]
        for v in columns:
            # Build the arguments for this row of the table
            args = defaults.copy()
            args[params['iterate_over']] = v
            # Blur the image
            blurred = main.blur(img, args['blur'], args['noise'])

            # Deblur the image accordingly with each method
            for i, method in enumerate(methods):
                phi_dphi = (main.methods[method[0]]['phi'], main.methods[method[0]]['dphi'])
                minFun = main.sci_minimize if method[1] == 'scipy' else main.our_minimize
                deblurred = main.phasen(blurred, args['lambda'], phi_dphi, minFun)
                table[i].append((deblurred[1],deblurred[2]))

        f = open('report/vars-' + params['iterate_over'] + '.tex', 'w+')
        labels = map(params['label'], columns)
        f.write(plot_table(table, labels))
        f.close()

'''
    Takes in a m \times n \times k matrix and plots it as an m \times n
    table where each cell shows a k-ary tuple
'''
def plot_table(tbl, y_labels=None):
    res = '\\begin{NiceTabular}{|' + ('c|' * len(tbl[0])) + '}\\hline\n'
    if y_labels != None:
        for cell in y_labels:
            res += cell + ' &'
        res = res[:-1] + '\\\\\\hline\\hline\n'

    for row in tbl:
        for cell in row:
            res += ' $' + print_tuple(cell) + '$ &'
        res = res[:-1] + '\\\\\\hline\n' # remove the last & and add \\\hline + newline
    return res+'\\end{NiceTabular}'

'''
    Prints a n-ary tuple rounding values appropriately
    NOTE: assumes all values are number-like and can be applied to round()
'''
def print_tuple(tpl):
    res='('
    for item in tpl:
        res += str(round(item, decimal_places)) + ','
    return res[:-1]+')'

actions = {
    'methods': plot_methods,
    'vars': plot_vars
}

if __name__ == '__main__':
    args = sys.argv[1:]
    if(len(args) == 0 or args[0] not in actions):
        print('Esempio di esecuzione:\npython plot.py [methods|vars]')
        exit()

    action = args[0]
    plt_common()
    actions[action]()
