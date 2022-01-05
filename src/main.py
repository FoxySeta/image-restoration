import lib
import numpy as np
from skimage import metrics
from scipy.optimize import minimize as sciminimize
import matplotlib.pyplot as plt

# tweak me
MAXITER = 10

# Array of (kernel length, standard deviation)
blurs = ((5, 0.5), (7, 1), (9, 1.3))

def phase1(path, plot=True):
    img = plt.imread(path).astype(np.float64)
    bs = []
    for blur in blurs:
        # Generate a blurring filter
        K = lib.psf_fft(lib.gaussian_kernel(*blur), blur[0], img.shape)
        # Generate noise
        noise = np.random.normal(size=img.shape) * 0.05
        # Apply blur and noise
        b = lib.A(img, K) + noise
        PSNR = metrics.peak_signal_noise_ratio(img, b)
        bs.append((b, PSNR, K))

    # plot the original image alonside three differently-blurred ones
    if plot:
        plt.figure(figsize=(30, 10))
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title('immagine originale')
        for i in range(3):
            ax2 = plt.subplot(2, 2, 2+i)
            ax2.imshow(bs[i][0], cmap='gray', vmin=0, vmax=1)
            plt.title(f'immagine corrotta con sigma {blurs[i]} (psnr: {bs[i][1]:.2f})')
        plt.show()

    return bs
    
def next_step(x,f,df): # backtracking procedure for the choice of the steplength
    alpha=1.1
    rho = 0.5
    c1 = 0.25
    p=-df
    j=0
    jmax=10
    while((f(x+alpha*p) > f(x)+c1*alpha*df.T@p) and j<jmax):
        alpha= rho*alpha
        j+=1
    if (j>jmax):
        return -1
    else:
        #print('alpha=',alpha)
        return alpha

def minimize(x0,b,f,df,mode,maxiter,abs_stop) -> np.ndarray: # funzione che implementa il met odo del gradiente
    #declare x_k and gradient_k vectors
    if mode=='plot_history':
        x=np.zeros((x0.size,maxiter))
    norm_grad_list=np.zeros((1,maxiter))
    function_eval_list=np.zeros((1,maxiter))
    error_list=np.zeros((1,maxiter))
    #initialize first values
    x_last = np.zeros((x0.size))
    print(x_last.shape)
    if mode=='plot_history':
        x[:,0] = x_last
    k=0
    function_eval_list[:,k]=f(x_last)
    error_list[:,k]=np.linalg.norm(x_last-b)
    norm_grad_list[:,k]=np.linalg.norm(df(x_last))
    while (np.linalg.norm(df(x_last))>abs_stop and k < maxiter-1):
        k=k+1
        grad = df(x_last)#direction is given by gradient of the last iteration
        # backtracking step
        step = next_step(x_last,f,grad)
        # Fixed step
        #step = 0.1
        if(step==-1):
            print('non convergente')
            return (k) #no convergence
        x_last=x_last-step*grad
        if mode=='plot_history':
            x[:,k] = x_last
        function_eval_list[:,k]=f(x_last)
        error_list[:,k]=np.linalg.norm(x_last-b)
        norm_grad_list[:,k]=np.linalg.norm(df(x_last))
    function_eval_list = function_eval_list[:,:k+1]
    error_list = error_list[:,:k+1]
    norm_grad_list = norm_grad_list[:,:k+1]
    print('iterations=',k)
    if mode == 'plot_history':
        print('last guess: x=(%f,%f)'%(x[0,k],x[1,k]))
    return x_last

def phasen(blurred, generate_functions, minimize):
    b = blurred[0]
    f, df = generate_functions(blurred[2],b)
    return minimize(np.zeros(b.shape), b, f, df)

# l = lambda
# regulating_factor = fn(X) -> matrix
# regulating_factor_grad = fn(X) -> matrix
def f_generator(l, regulating_term, regulating_term_grad):
    def hardcode_f(K,b):
        def f(x):
            X = x.reshape(b.shape)
            res = 0.5*(np.linalg.norm(lib.A(X, K)-b)**2) + l*regulating_term(X)
            return np.sum(res)
        def df(x):
            X = x.reshape(b.shape)
            res = lib.AT(lib.A(X, K)-b, K) + l*regulating_term_grad(X)
            newRes = np.reshape(res, b.size)
            return newRes
        return (f,df)
    return hardcode_f

def our_minimize(x0, b, f, df):
    return np.reshape(minimize(np.reshape(x0, b.size), np.reshape(b, b.size), f, df, 'plot_history',MAXITER,MAXITER*2), b.shape)

def sci_minimize(x0, _, f, df):
    return np.reshape(sciminimize(f, x0, method='CG', jac=df, options={'maxiter':MAXITER}).x, x0.shape)

# files = ['1', '2', '3', '4', '5', '6', '7', '8', 'A', 'B']
files = ['4']
lambdas = [0.02,0.04,0.06]
for file in files:
    blurred_images = phase1('img/' + file + '.png', False)
    deblurred = []
    for l in lambdas:
        naive_fns = f_generator(0, lambda _: 0, lambda _: 0)
        tikhonov_fns = f_generator(l, lambda X: 0.5*np.linalg.norm(X)**2, lambda X: X)
        tv_fns = f_generator(l, lib.totvar, lib.grad_totvar)
        for blurred in blurred_images:
            print(blurred[2].shape)
            deblurred_image = phasen(blurred, tv_fns, sci_minimize) #our_minimize)
            deblurred.append(deblurred_image)

    plt.figure(figsize=(21, 7))
    plt.suptitle(f'Deblur con {MAXITER} iterazioni')
    lines = 1+len(lambdas)
    for i in range(3):
        ax1 = plt.subplot(lines, len(blurred_images), i+1)
        ax1.imshow(blurred_images[i][0], cmap='gray', vmin=0, vmax=1)
        plt.title('immagine corrotta')

    for i in range(len(lambdas)):
        for j in range(len(blurred_images)):
            ax2 = plt.subplot(lines, len(blurred_images), lines+(i*len(blurred_images))+j)
            ax2.imshow(deblurred[i*len(blurred_images)+j], cmap='gray', vmin=0, vmax=1)
            plt.title(f'immagine ripristinata con lambda={lambdas[i]}')
    plt.show()
