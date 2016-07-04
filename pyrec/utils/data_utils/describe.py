from collections import defaultdict
import numpy as np
import tabulate
import pylab as plt


class Describe(object):

    def __init__(self, data_matrix):
        super(Describe, self).__init__()
        self.data_matrix = data_matrix
        self.m , self.n = self.data_matrix.shape
        
    def getAggregateStats(self, axis=0):
#         assert self.data_matrix.max() == 1.0
        counts = np.ravel(self.data_matrix.sum(axis=axis))
        return counts
    
    def aggregateBycount(self, axis=0):
        count_map = defaultdict(int)
        counts = self.getAggregateStats(axis)
        for count in counts:
            count_map[count] += 1
        return count_map
    
    def get_num_users(self):
        return self.m
    
    def get_num_items(self):
         return self.n

    def getSortedKeyVals(self, count_map):
        key_val = count_map.items()
        key_val.sort(key=lambda x: x[0])
        keys,val = zip(*key_val)
        return keys,val
   
    
    def describe(self, plot=True, save_plot=False, 
                 interaction="Purchase",
                 plot_style="ggplot",
                 save_path= "fig.pdf", 
                 save_format = "pdf"):
        def fixTicksSize(ax):
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.tick_params(axis='both', which='minor', labelsize=16)
        plt.style.use(plot_style)

        user_stats = self.aggregateBycount(axis=1)
        ukey, ucount = self.getSortedKeyVals(user_stats)
        item_stats = self.aggregateBycount(axis=0)
        ikey, icount = self.getSortedKeyVals(item_stats)
        #add users and items count
        data = []
        dash = "-"
        data.append(["count", self.m, self.n, self.data_matrix.data.sum()])
        
        ustat = np.ravel(self.data_matrix.sum(axis=1))
        istat = np.ravel(self.data_matrix.sum(axis=0))
        
        data.append(["mean", np.mean(ustat), np.mean(istat), dash])
        data.append(["std",np.std(ustat), np.std(istat), dash])
        data.append(["median",np.median(ustat), np.median(istat), dash])
        
        print tabulate.tabulate(data, 
                                headers=["","Users", "Items", interaction],
                                tablefmt='orgtbl',
                               numalign ="left")
        
        fig = plt.figure(figsize=(15,9))
        ax1 = fig.add_subplot(1,2,1)
        fixTicksSize(ax1)
        ax1.scatter(ukey,ucount, alpha=0.8, rasterized=False)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_ylabel("#Users",fontsize=25)
        ax1.set_xlabel("#" + interaction,fontsize=25)
        ax1.set_title("#Users Vs #" + interaction,fontsize=25)

        ax2 = fig.add_subplot(1,2,2)
        fixTicksSize(ax2)
        ax2.scatter(ikey,icount, alpha=0.8, rasterized=False)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("#" + interaction,fontsize=25)
        ax2.set_ylabel("#Items",fontsize=25)
        ax2.set_title("#Items vs #" + interaction,fontsize=25)

        fig.tight_layout()
        
        if save_plot:
            fig.savefig(save_path, format=save_format)
        


if __name__ == "__main__":
    import sys
    from pyrec.utils.data_utils.data import loadDataset, Data
    from pyrec.utils.data_utils.lineParser import userItemRatingParser
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    delim = sys.argv[3]
    save_path = sys.argv[4]
    save_format = sys.argv[5]

    parser = userItemRatingParser(delim=delim)
    train, test, _ = getTrainTest(train_path, test_path, parser)
    describe = Describe(train)
    describe.describe(
        save_plot=True, save_path=save_path, save_format=save_format)
