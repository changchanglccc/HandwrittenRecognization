# Calculates the new centroid to a cluster
def updateMeans(means, clusters):
    maxDist = -1
    for i, cluster in enumerate(clusters):
        # create empty centroid
        newCentroid = [0.0]*784
        # Calculate center point by adding everything and diving by the number of digits
        for (label, digit) in cluster:
            newCentroid = list(np.array(newCentroid) + np.array(digit))
        newCentroid = list(np.array(newCentroid) * (1.0 /(len(cluster) if len(cluster) > 0 else 1.0)))
        # calculate the distance from the old to the new centroid
        dist = 0.0
        if len(newCentroid) != 0:
            dist = distance(means[i], newCentroid)
        if dist > maxDist:
            maxDist = dist
        means[i] = newCentroid
    # Return biggest distance moved
    return maxDist

# Calculates the distance between 2 point in any dimmession
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Get the cluster index with the smallest distance to a given digit
def getBestDistanceCluster(means, digit):
    minDist = 1000000.0
    bestIndex = -1
    for i, mean in enumerate(means):
        dist = distance(digit, mean)
        if minDist > dist:
            minDist = dist
            bestIndex = i
    return bestIndex

# Classify digits according to clusters
def testKMeans(means, clusters):
    correct = 0
    wrong = 0
    clusterChosen = getBestDistanceCluster(means, testingList[0])
    for i, test in enumerate(testingList):
        clusterChosen = getBestDistanceCluster(means, test)
        if clusters[clusterChosen][0] == testingLabels[i]:
            correct += 1
        else:
            wrong += 1
    print("Correct test", (correct/len(testingList))*100,"%")
    print("wrong Tests", (wrong/len(testingList))*100, "%")

# Assigns a Label to a cluster using the most seen label
def labelCluster(cluster):
    N_DIGITS = 10
    n = [0 for c in range(N_DIGITS)]
    for (label, digit) in cluster:
        n[label] += 1
    return n.index(max(n))

# K means clustering for digit recognition
def k_means(k, trainingList):
    print("Starting K-Means Clustering with "+str(k)+ " clusters")
    # randomly select k centroid
    means = []
    for i in range(k):
        means.append(random.sample(trainingList, 1)[0][1])
    rep = False
    change = True
    meanDistChange = 10000.0
    minMeanChange = 100.0
    clusters = []
    while True:
        # if means need to be recalculated
        if rep: meanDistChange = updateMeans(means, clusters)
        # if the max distance moved was smaller than the threshold, stop clustering
        if meanDistChange <= minMeanChange:
            break
        # clean clusters
        clusters = [[] for c in range(k)]
        # For every training digit
        for i, (label, feat) in enumerate(trainingList):
            # Choose best cluster to go
            clusterChosen = getBestDistanceCluster(means, feat)
            # Assign to that cluster
            clusters[clusterChosen].append((label, feat))
            update_progress(i, len(trainingList))
        rep = True
        print(" -> max distance moved = "+ str(meanDistChange)+ ", threshold = "+str(minMeanChange))
    # Assign Labels to clusters
    for i, cluster in enumerate(clusters):
        clusters[i] = (labelCluster(cluster), cluster)
    # Show details for each cluster
    showClusteringDetails(clusters)
    # Test against test data
    testKMeans(means, clusters)

# Shows digits in each cluster class and the label trained 
def showClusteringDetails(clusters):
    print("| Ck [Label=n]| ", list(range(10)), "|")
    print("==================================================")
    for i, (label, cluster) in enumerate(clusters):
        print("| C"+str(i)+" [Label="+str(label)+"]| ", end="")
        k = [0 for c in range(10)]
        for (label2, digit) in cluster:
            k[label2] += 1
        print(k, " |")
    print("============================================")