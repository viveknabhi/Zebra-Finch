from sklearn.cluster import AgglomerativeClustering
import accuracy

def Hierarchical_Cluster(data,n_clusters,metric):
    model = AgglomerativeClustering(n_clusters)
    labels = model.fit_predict(data)

    score = accuracy.getAccuracy(data,labels,len(data),metric)
    return(score)
    
    
