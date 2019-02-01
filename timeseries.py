from fastai.vision import *

__all__ = ["TimeSeriesItem","TimeSeriesList","UCRArchive"]

class TimeSeriesItem(ItemBase):
    def __init__(self,ts,name=""):
        self.data = self.obj = torch.tensor(ts,dtype=torch.float)
        self.name = name
        
    def show(self, ax, title="",channels=None, **kwargs):
        x = self.data
        if channels: x = x[:,channels]
        ax.plot(x)
        ax.set_title(title)
        
    def show_channels(self):
        rows = math.ceil(self.data.shape[1] / 3)
        fig, axs = plt.subplots(rows,3,figsize=(18,16))
        for i, ax in enumerate(axs.flatten()):
            ax.plot(self.data[:,i])
            ax.set_title(f"Channel {i}")
    
    def __str__(self):
        return f"Time series of size {self.data.shape}"

class TimeSeriesList(ItemList):
    def __init__(self,items,labeled=-1,**kwargs):
        self.labeled = labeled
        super().__init__(items, **kwargs)
    
    def new(self, items, **kwargs):
        return super().new(items,labeled=self.labeled,**kwargs)
    
    @classmethod
    def from_csv_list(cls,csv_paths,labelCol=-1,header=None,sep="\t",**kwargs):
        dfs = []
        nl = []
        for f in csv_paths:
            dfs.append(pd.read_csv(f,header=header,sep=sep))
            nl += [f.name]*len(dfs[-1])
        df = pd.concat(dfs)
        return cls(df.values,labeled=labelCol,xtra=np.array(nl))        
    
    @classmethod
    def from_numpy(cls,nparray,**kwargs):
        return cls(items=nparray,**kwargs)
    
    def get(self, i):
        a = super().get(i).astype(np.float)
        return TimeSeriesItem(np.delete(a,self.labeled,0) if self.labeled >= 0 else a)
    
    def get_class(self,i):
        a = super().get(i).astype(np.float)
        return a[self.labeled] if self.labeled >= 0 else 0
    
    def split_by_csv_name(self, valid_names):
        return self.split_by_idx(np.where(np.isin(self.xtra,valid_names))[0])
    
    def label_from_col(self, **kwargs):
        return self.label_from_func(lambda o: o[self.labeled],label_cls=CategoryList)
    
    def label_from_self(self, **kwargs):
        return self._label_list(x=self,y=self)
        
    def reconstruct(self, t):
        if isinstance(t,list):
            t = t[0]
        return TimeSeriesItem(t.numpy())
    
    def show_xys(self, xs, ys, figsize=(12,10),channels=None,plotY=True, **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].show(ax=ax,channels=channels)
            if plotY and isinstance(ys[0],TimeSeriesItem):
                ys[i].show(ax=ax)
                
    def show_xyzs(self, xs, ys, zs, attn=None,channels=None,figsize=(12,10), **kwargs):
        rows = int(math.sqrt(len(xs)))
        fig, axs = plt.subplots(rows,rows,figsize=figsize)
        for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
            xs[i].show(ax=ax,channels=channels)
            if isinstance(zs[0],TimeSeriesItem):
                zs[i].show(ax=ax)
            if attn:
                ax.plot(10*attn.attnSave[i].flatten())
                
# Simple utility class to manage the datasets in the UCR archive, datasets available for download here: 
# https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

class UCRArchive():
    def __init__(self,path=Path("UCRArchive_2018")):
        self.path = path if isinstance(path,Path) else Path(path)
        self.datasets = [f for f in self.path.iterdir() if f.is_dir()]
        self.dsdict = {f.name : f for f in self.datasets}
        
    def get(self,name):
        f = self.dsdict[name]
        return f
    
    def get_csv_files(self,name):
        f = self.dsdict[name]
        return list(f.glob("*.tsv"))
    
    def list_datasets(self):
        return [d.name for d in self.datasets]
    
    def category_distribution(self,name):
        path = self.dsdict[name]
        trainDF,testDF = [pd.read_csv(p,header=None,sep="\t") for p in [path/f"{name}_TRAIN.tsv",path/f"{name}_TEST.tsv"]]
        trainDF.iloc[:,0].hist(alpha=0.3)
        testDF.iloc[:,0].hist(alpha=0.3)
        return trainDF.iloc[:,0].value_counts()