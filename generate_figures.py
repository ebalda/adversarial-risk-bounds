import numpy as np
import matplotlib.pyplot as plt
import os
import tikzplotlib

# Global dictionaries
summary_dict = {'adv-training': 0,
                'std-train-error': 1,
                'adv-train-error': 2,
                'std-test-error': 3,
                'adv-test-error': 4
                }
norm_dict = {
    'mixed-half-inf': 0,
    'mixed-1-inf': 1,
    'mixed-1-1': 2,
    'fro': 3,
    'spectral': 4,
    'mixed-1-2': 5,
    'mixed-2-1': 6
}


def get_gbound(norm_mtx_list, norm_dict, bound='ours'):
    gbound = 0.

    if bound=='ours':
        for jj in range(3):
            norms_mtx = norm_mtx_list[jj].copy()
            norms_mtx = np.diag(1/norms_mtx[:, norm_dict['mixed-1-inf']]) @ norms_mtx # rebalance

            gbound += np.sqrt(norms_mtx[:, norm_dict['mixed-1-1']]*norms_mtx[:, norm_dict['mixed-half-inf']])


    elif bound=='tu':
        for jj in range(3):
            norms_mtx = norm_mtx_list[jj].copy()
            norms_mtx = np.diag(1/norms_mtx[:, norm_dict['spectral']]) @ norms_mtx # rebalance

            gbound +=  np.sqrt(norms_mtx[:, norm_dict['fro']])

        gbound = gbound**2.
    elif bound=='khim':
        fro_vals = np.zeros([norm_mtx_list[0].shape[0], 3])
        for jj in range(3):
            norms_mtx = norm_mtx_list[jj].copy()
            norms_mtx = np.diag(1/norms_mtx[:, norm_dict['mixed-1-inf']]) @ norms_mtx # rebalance
            fro_vals[:,jj] = norms_mtx[:, norm_dict['fro']]


        gbound = np.max(fro_vals,axis=1)

    elif bound=='barlett':
        for jj in range(3):
            norms_mtx = norm_mtx_list[jj].copy()
            norms_mtx = np.diag(1/norms_mtx[:, norm_dict['spectral']]) @ norms_mtx # rebalance

            gbound += (norms_mtx[:, norm_dict['mixed-2-1']])**(2/3)
        gbound = gbound**(3/2)
    elif bound=='neyshabur':
        for jj in range(3):
            norms_mtx = norm_mtx_list[jj].copy()
            norms_mtx = np.diag(1/norms_mtx[:, norm_dict['spectral']]) @ norms_mtx # rebalance

            gbound += (norms_mtx[:, norm_dict['fro']])**2.

        gbound = np.sqrt(gbound)
    return gbound


def plot_bounds(model='fcnnmnist', legend=True):

    summary_mtx = np.loadtxt('./results/'+model+'-summary.csv', delimiter=';')

    norm_mtx_list = []
    for jj in range(3):
        norm_mtx_list.append(np.loadtxt('./results/'+model+'-W_{:.0f}.csv'.format(jj + 1), delimiter=';'))

    # Get start of adversarial training
    time = np.linspace(0,100,len(summary_mtx))
    idx = np.argmax(summary_mtx[:,summary_dict['adv-training']]).min()
    adv_start_time = time[idx-1]

    # Start plotting
    legend_handels = []
    legend_labels = []
    plt.rc('text', usetex=True)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.grid()



    marker_step=4
    hdl, = ax.plot(time[:-1], summary_mtx[:-1, summary_dict['std-train-error']], '-b', markersize=6, fillstyle='none')
    legend_handels.append(hdl)
    legend_labels.append('Train error')

    hdl, = ax.plot(time[:-1:marker_step], summary_mtx[:-1:marker_step, summary_dict['std-test-error']], 'ob',
            markersize=6, fillstyle='none')
    legend_handels.append(hdl)
    legend_labels.append('Test error')

    hdl, = ax.plot(time[:-1], summary_mtx[:-1, summary_dict['adv-train-error']], '--r', markersize=6, fillstyle='none')
    legend_handels.append(hdl)
    legend_labels.append('Train error (adv)')

    hdl, = ax.plot(time[:-1:marker_step], summary_mtx[:-1:marker_step, summary_dict['adv-test-error']] , 'vr',
            markersize=6, fillstyle='none')
    legend_handels.append(hdl)
    legend_labels.append('Test error (adv)')

    # leg = ['train err', 'test err', 'train err (adv)', 'train err (adv)']
    bound_colors = ['g', 'c', 'm', 'g', 'k']
    bound_markers = ['>', '<', 'D', '^', '*']
    bound_names = ['barlett', 'neyshabur', 'tu', 'khim', 'ours']
    bound_labels = {
        'barlett':r'Barlett \cite{Bartlett2017SpectrallynormalizedMB}',
        'neyshabur':r'Neyshabur \cite{neyshabur2017pac}',
        'tu':r'Tu \cite{Tu2018TheoreticalAO} (adv)',
        'khim': r'Khim \cite{Khim2018AdversarialRB} (adv)',
        'ours':r'Ours (adv)'
    }

    marker_step = 8
    for jj in range(len(bound_names)):
        print(bound_names[jj])
        gbound = get_gbound(norm_mtx_list, norm_dict, bound=bound_names[jj])
        hdl, = ax.plot(time[2*jj:-1:marker_step],
                2.*gbound[2*jj:-1:marker_step]/gbound[:-1].max(),
                bound_colors[jj] + bound_markers[jj],
                markersize=6, fillstyle='none')
        legend_handels.append(hdl)
        legend_labels.append(bound_labels[bound_names[jj]])

        gbound = get_gbound(norm_mtx_list, norm_dict, bound=bound_names[jj])
        ax.plot(time[:-1],
                2. *gbound[:-1]/gbound[:-1].max(),
                bound_colors[jj] + '-',
                markersize=6, fillstyle='none')




    ax.plot([adv_start_time, adv_start_time], [0, 2], '-.k')
    ax.set_xlim(-1,101)

    plt.xticks([0,25,50,75,100])
    ax.tick_params(labelsize=18)
    ax.set_xlabel(r'Training Time (in \%)', fontsize=18)
    plt.tight_layout()

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    if legend:
        ax.annotate('start of \n adversarial \n training',
                    xy=(adv_start_time - 50, 1.1),
                    xytext=(adv_start_time - 50, 1.1),
                    fontsize=16,
                    annotation_clip=False)
        # Put a legend to the right of the current axis
        ax.legend(legend_handels, legend_labels,
                  loc='center left', framealpha=0.99,
                  fontsize=14,
                  bbox_to_anchor=(1, 0.5))

    fname = './figures/'+model+'-bound.pdf'

    plt.savefig(fname, format='pdf')
    os.system('pdfcrop {} {}'.format(fname, fname))

    tikzplotlib.save(fname.split('figures/')[0] + 'figures/tikz/' +  fname.split('figures/')[1].split('.pdf')[0] + '.tikz')

    # plt.show()
    plt.close()

def plot_norms(model='fcnnmnist', legend=True):
    summary_mtx = np.loadtxt('./results/' + model + '-summary.csv', delimiter=';')
    norm_mtx_list = []
    for jj in range(3):
        norm_mtx_list.append(np.loadtxt('./results/'+model+'-W_{:.0f}.csv'.format(jj + 1), delimiter=';'))

    # Get start of adversarial training
    time = np.linspace(0,100,len(summary_mtx))
    idx = np.argmax(summary_mtx[:,summary_dict['adv-training']]).min()
    adv_start_time = time[idx-1]

    for jj in range(3):
        norms_mtx = norm_mtx_list[jj].copy()

        # Plot effective joint sparsity
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.grid()
        s1 = norms_mtx[:, norm_dict['mixed-half-inf']] / norms_mtx[:, norm_dict['mixed-1-inf']]
        s2 = norms_mtx[:, norm_dict['mixed-1-1']] / norms_mtx[:, norm_dict['mixed-1-inf']]

        maxy = np.maximum(s1[:-1].max(), s2[:-1].max())

        ax.plot(time[:-1], s1[:-1], '-.r', markersize=6, fillstyle='none')
        ax.plot(time[:-1], s2[:-1], '-b', markersize=6, fillstyle='none')


        ax.plot([adv_start_time, adv_start_time], [0, maxy+10], '-.k')


        ax.set_xlim(-1, 101)
        ax.set_xlabel(r'Training Time (in \%)', fontsize=20)
        ax.tick_params(labelsize=20)
        # ax.set_title(r'Layer {:.0f}'.format(jj+1), fontsize=20)

        if legend and jj==0:
            ax.annotate('start of \n adversarial \n training',
                        xy=(adv_start_time -50, maxy/2),
                        xytext=(adv_start_time -50, maxy/2),
                        fontsize=18,
                        annotation_clip=False)
            leg = [r'$\bar s_1$', r'$\bar s_2$']
            plt.legend(leg, framealpha=0.7, fontsize=18)

        fname = './figures/'+model+'-norms-W_{:.0f}.pdf'.format(jj+1)
        plt.tight_layout()
        plt.savefig(fname, format='pdf')
        os.system('pdfcrop {} {}'.format(fname, fname))

        tikzplotlib.save(fname.split('figures/')[0] + 'figures/tikz/' +fname.split('figures/')[1].split('.pdf')[0]+'.tikz')
        plt.close()




def main():
    # Create necesary directories
    if not os.path.exists('./figures/'):
        os.makedirs('./figures/')

    if not os.path.exists('./figures/tikz/'):
        os.makedirs('./figures/tikz/')

    fname = 'figslist.txt'
    # os.system('ls ./results/ID-0*fcnnmnist*summary.csv > ' + fname)
    os.system('ls ./results/ID-59903*fcnncifar*summary.csv > ' + fname)


    with open(fname) as f:
        lines = f.readlines()
        for ii in range(0, len(lines)):
            ID = lines[ii].split('ID-')[1].split('_')[0]
            model = lines[ii].split('_')[1].split('-')[0]
            if model=='fcnnmnist':
                legflag = True
            else:
                legflag=False
            plot_bounds(model='ID-{}_{}'.format(ID,model), legend=legflag)
            plot_norms(model='ID-{}_{}'.format(ID, model), legend=legflag)


    return




if __name__ == '__main__':
    main()

