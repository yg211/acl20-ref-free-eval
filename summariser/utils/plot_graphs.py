import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from summariser.utils.misc import getSoftmaxProb

def plotGapAgreementRatio(x,y,bin_num=10):
    minx = min(x)
    bin_width = (max(x)-minx)/bin_num

    y_acc = [0]*bin_num
    for idx in range(len(x)):
        ii = (x[idx]-minx)
        ii /= bin_width
        if ii >= bin_num:
            ii = bin_num-1
        y_acc[int(ii)] += y[idx]

    x_values = np.linspace(minx,max(x),bin_num)
    plt.plot(x_values,y_acc)

def plotDistribution(rouge_list,new_list):
    sns.distplot(rouge_list,1000,label='rouge')
    sns.distplot(new_list,1000,label='synthetic')
    plt.legend()
    plt.show()

def plotCumSampleProbDiff(rouge_list,new_list,strict=1.):
    rouge_prob = getSoftmaxProb(rouge_list,strict)
    syn_prob = getSoftmaxProb(new_list,strict)

    corresponding_syn_prob = []
    sorted_rouge_prob = sorted(rouge_prob)
    for ele in sorted_rouge_prob:
        corresponding_syn_prob.append(syn_prob[sorted_rouge_prob.index(ele)])

    cum_rouge_prob = np.cumsum(sorted_rouge_prob)
    cum_syn_prob = np.cumsum(corresponding_syn_prob)

    plt.plot(cum_syn_prob,label='synthetic cumulative prob',alpha=0.6)
    plt.plot(cum_rouge_prob,label='rouge cumulative prob',alpha=0.9,color='red')
    plt.legend()
    plt.show()


def plotSampleProbDiff(rouge_list,new_list,strict=1.):
    rouge_prob = getSoftmaxProb(rouge_list,strict)
    syn_prob = getSoftmaxProb(new_list,strict)
    plotDiff(rouge_prob,syn_prob)

def plotSampleProbBars(rouge_list,new_list,strict=1.,bin_num=100):
    rouge_prob = getSoftmaxProb(rouge_list,strict)
    syn_prob = getSoftmaxProb(new_list,strict)

    corresponding_syn_prob = []
    sorted_rouge_prob = sorted(rouge_prob)
    for ele in sorted_rouge_prob:
        corresponding_syn_prob.append(syn_prob[sorted_rouge_prob.index(ele)])

    bin_size = int(len(rouge_prob)/bin_num)
    rouge_bar = []
    syn_bar = []

    pointer = 0
    for bin_idx in range(bin_num):
        ll = sorted_rouge_prob[pointer:pointer+bin_size]
        rouge_bar.append(np.sum(ll))
        ll = syn_prob[pointer:pointer+bin_size]
        syn_bar.append(np.sum(ll))
        pointer += bin_size


    ind = np.arange(bin_num)
    width = 0.35
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(ind,syn_bar,width,label='synthetic')
    ax.bar(ind+width,rouge_bar,width, label='rouge')
    ax.legend()
    plt.show()

def plotDiff(rouge_list,new_list):
    sorted_true = sorted(rouge_list)
    corresponding_new = []

    for ele in sorted_true:
        corresponding_new.append(new_list[rouge_list.index(ele)])

    #sliding window smooth
    window_size = 500
    smooth = np.convolve(corresponding_new,np.ones(window_size,)/window_size,mode='valid')

    plt.plot(corresponding_new,label='synthetic',alpha=0.6)
    plt.plot(smooth,label='synthetic-smoothed',alpha=0.9,color='green')
    plt.plot(sorted_true,label='rouge',alpha=1.,color='red')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ll = [1,2,3,4,float('nan'),6,7]
    plt.plot(ll)
    plt.show()