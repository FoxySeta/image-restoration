def phase2(blurred_images, plot):
    deblurred = []
    for blurred in blurred_images:
        b = blurred[0]
        K = blurred[2]
        def f(x):
          X = x.reshape(b.shape)
          res = 0.5*(np.linalg.norm(lib.A(X, K)-b)**2)
          return np.sum(res)

        def df(x):
          X = x.reshape(b.shape)
          res = lib.AT(lib.A(X, K)-b, K)
          newRes = np.reshape(res, b.size)
          return newRes

        X = sciminimize(f, np.zeros(b.shape), method='CG', jac=df, options={'disp':True,'maxiter':MAXITER}).x
        deblurred.append(X.reshape(b.shape))

    # plot each blurred image with its deblurred counterpart on the bottom
    if plot:
        plt.figure(figsize=(21, 7))
        plt.suptitle(f'Deblur con {MAXITER} iterazioni')
        for i in range(3):
            ax1 = plt.subplot(2, 3, i+1)
            ax1.imshow(blurred_images[i][0], cmap='gray', vmin=0, vmax=1)
            plt.title('immagine corrotta')
            ax2 = plt.subplot(2, 3, i+4)
            ax2.imshow(deblurred[i], cmap='gray', vmin=0, vmax=1)
            plt.title('immagine ripristinata')
        plt.show()

def phase3(blurred_images, ours, plot):
    deblurred = []
    for blurred in blurred_images:
        for l in LAMBDAS:
            b = blurred[0]
            K = blurred[2]
            def f(x):
              X = x.reshape(b.shape)
              res = 0.5*(np.linalg.norm(lib.A(X, K)-b)**2) + l*0.5*np.linalg.norm(X)**2
              return np.sum(res)

            def df(x):
              X = x.reshape(b.shape)
              res = lib.AT(lib.A(X, K)-b, K) + l*X
              newRes = np.reshape(res, b.size)
              return newRes
            if ours:
                X = minimize(np.reshape(np.zeros(b.shape), b.size), np.reshape(b, b.size), f, df, 'plot_history',MAXITER,MAXITER*2)
            else:
                X = sciminimize(f, np.zeros(b.shape), method='CG', jac=df, options={'disp':True,'maxiter':MAXITER}).x
            deblurred.append(X.reshape(b.shape))

    # plot each blurred image with its deblurred counterpart on the bottom
    if plot:
        plt.figure(figsize=(21, 7))
        plt.suptitle(f'Deblur con {MAXITER} iterazioni')
        lines = 1+len(LAMBDAS)
        for i in range(3):
            ax1 = plt.subplot(3, lines, i+1)
            ax1.imshow(blurred_images[i][0], cmap='gray', vmin=0, vmax=1)
            plt.title('immagine corrotta')
            for j in range(len(LAMBDAS)):
                ax2 = plt.subplot(3, lines, 4+i+3*j)
                ax2.imshow(deblurred[i*len(LAMBDAS)+j], cmap='gray', vmin=0, vmax=1)
                plt.title(f'immagine ripristinata con lambda={LAMBDAS[j]}')
        plt.show()

