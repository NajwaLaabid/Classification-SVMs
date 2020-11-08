import csv
import numpy
import matplotlib.pyplot as plt
import math
import scipy.stats
from decimal import Decimal

VERBOSE_TREE = False

# Load a CSV file
def load_csv(filename, last_column_str=False, normalize=False, as_int=False):
    dataset = list()
    head = None
    classes = {}
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for ri, row in enumerate(csv_reader):
            if not row:
                continue
            if ri == 0:
                head = row
            else:
                rr = [r.strip() for r in row]
                if last_column_str:
                    if rr[-1] not in classes:
                        classes[rr[-1]] = len(classes)
                    rr[-1] = classes[rr[-1]]
                dataset.append([float(r) for r in rr])
    dataset = numpy.array(dataset)
    if not last_column_str and len(numpy.unique(dataset[:,-1])) <= 10:
        classes = dict([("%s" % v, v) for v in numpy.unique(dataset[:,-1])])
    if normalize:
        dataset = normalize_dataset(dataset)
    if as_int:
        dataset = dataset.astype(int)
    return dataset, head, classes

# Find the min and max values for each column
def dataset_minmax(dataset):
    return numpy.vstack([numpy.min(dataset, axis=0), numpy.max(dataset, axis=0)])

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax=None):
    if minmax is None:
        minmax = dataset_minmax(dataset)
    return (dataset - numpy.tile(minmax[0, :], (dataset.shape[0], 1))) / numpy.tile(minmax[1, :]-minmax[0, :], (dataset.shape[0], 1))

# Sample k random points from the domain 
def sample_domain(k, minmax=None, dataset=None):
    if dataset is not None:
        minmax = dataset_minmax(dataset)
    if minmax is None:
        return numpy.random.random(k)
    d = numpy.random.random((k, minmax.shape[1]))
    return numpy.tile(minmax[0, :], (k, 1)) + d*numpy.tile(minmax[1, :]-minmax[0, :], (k, 1))

# Compute distances between two sets of instances
def euclidean_distance(A, B):
    return numpy.vstack([numpy.sqrt(numpy.sum((A - numpy.tile(B[i,:], (A.shape[0], 1)))**2, axis=1)) for i in range(B.shape[0])]).T

def L1_distance(A, B):
    return numpy.vstack([numpy.sum(numpy.abs(A - numpy.tile(B[i,:], (A.shape[0], 1))), axis=1) for i in range(B.shape[0])]).T

# Calculate contingency matrix
def contingency_matrix(actual, predicted, weights=None):
    if weights is None:
        weights = numpy.ones(actual.shape[0], dtype=int)
    ac_int = actual.astype(int)
    prd_int = predicted.astype(int)
    counts = numpy.zeros((numpy.maximum(2,numpy.max(prd_int)+1), numpy.maximum(2,numpy.max(ac_int)+1), 2), dtype=type(weights[0]))
    for p,a,w in zip(prd_int, ac_int, weights):
        counts[p, a, 0] += 1
        counts[p, a, 1] += w
    return counts

# Calculate metrics from confusion matrix
def TPR_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[1,1])/float(confusion_matrix[1,1]+confusion_matrix[0,1])
def TNR_CM(confusion_matrix):
    if confusion_matrix[0,0] == 0: return 0.
    return (confusion_matrix[0,0])/float(confusion_matrix[0,0]+confusion_matrix[1,0])
def FPR_CM(confusion_matrix):
    if confusion_matrix[1,0] == 0: return 0.
    return (confusion_matrix[1,0])/float(confusion_matrix[0,0]+confusion_matrix[1,0])
def FNR_CM(confusion_matrix):
    if confusion_matrix[0,1] == 0: return 0.
    return (confusion_matrix[0,1])/float(confusion_matrix[0,1]+confusion_matrix[1,1])
def recall_CM(confusion_matrix):
    return TPR_CM(confusion_matrix)
def precision_CM(confusion_matrix):
    if confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[1,1])/float(confusion_matrix[1,0]+confusion_matrix[1,1])
def accuracy_CM(confusion_matrix):
    if confusion_matrix[0,0] == 0 and confusion_matrix[1,1] == 0: return 0.
    return (confusion_matrix[0,0]+confusion_matrix[1,1])/float(confusion_matrix[0,0]+confusion_matrix[1,1]+
    confusion_matrix[0,1]+confusion_matrix[1,0])
metrics_cm = {"TPR": TPR_CM, "TNR": TNR_CM, "FPR": FPR_CM, "FNR": FNR_CM,
              "recall": recall_CM, "precision": precision_CM, "accuracy": accuracy_CM}
    
def get_CM_vals(actual, predicted, weights=None, vks=None):
    if vks is None:
        vks = metrics_cm.keys()
    cm = contingency_matrix(actual, predicted, weights)
    if weights is None:
        cm = cm[:, :, 0]
    else:
        cm = cm[:, :, 1]
    vals = {}
    for vk in vks:
        if vk in metrics_cm:
            vals[vk] = metrics_cm[vk](cm)
    return vals, cm


# Calculate the error rate for a split dataset
def error_rate(groups, lbls, weights=None):
    er = 0
    nb_instances = 0.
    for g in groups:
        if len(g) > 0:
            if weights is not None:
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cs = numpy.bincount(idx, minlength=len(weights))*weights.astype(float)
                nb_winstances = numpy.sum(cs)
                er += nb_winstances-numpy.max(cs)
                nb_instances += nb_winstances
            else:
                nb_instances += len(g)
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cis = numpy.bincount(idx)
                er += len(g)-numpy.max(cis)
    return er/nb_instances

# Calculate the entropy for a split dataset
def entropy(groups, lbls, weights=None):
    entropy = 0.
    nb_instances = 0.
    for g in groups:
        if len(g) > 0:
            if weights is not None:
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cs = numpy.bincount(idx, minlength=len(weights))*weights.astype(float)
                nb_winstances = numpy.sum(cs)
                pis = cs/nb_winstances
                nb_instances += nb_winstances
                entropy -= nb_winstances*numpy.sum(pis*numpy.log2(pis))
            else:
                nb_instances += len(g)
                ### ...
    return entropy/nb_instances

# Calculate the entropy for a split dataset
def information_gain(groups, lbls, weights=None):
    ce = entropy(groups, lbls, weights)
    pe = entropy([numpy.hstack(groups)], lbls, weights)
    return -(pe-ce)

# Calculate the Gini index for a split dataset
def gini(groups, lbls, weights=None):
    gini = 0.
    nb_instances = 0.
    for g in groups:
        if len(g) > 0:
            if weights is not None:
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                cs = numpy.bincount(idx, minlength=len(weights))*weights.astype(float)
                nb_winstances = numpy.sum(cs)
                pis = cs/nb_winstances
                nb_instances += nb_winstances
                gini += nb_winstances*(1 - numpy.sum(pis**2))
            else:
                nb_instances += len(g)
                u, idx = numpy.unique(lbls[g], return_inverse=True)
                pis = numpy.bincount(idx)/float(idx.shape[0])
                gini += idx.shape[0]*(1 - numpy.sum(pis**2))
    return gini/nb_instances

# Calculate accuracy percentage
def accuracy_metric(actual, predicted, weights=None):
    if weights is not None:
        ac_int = actual.astype(int)
        return numpy.sum(weights[ac_int[actual == predicted]])/float(numpy.sum(weights[ac_int]))
    return numpy.sum(actual == predicted)/float(actual.shape[0])

def split_dataset(dataset, ratio_train):
    ratio_train = ratio_train
    
    ids = numpy.random.permutation(dataset.shape[0])
    split_pos = int(len(ids)*ratio_train)
    train_ids, test_ids = ids[:split_pos], ids[split_pos:]
    train_set = dataset[train_ids]
    test_set = dataset[test_ids]

    return train_set, test_set

def get_thresholds(scores):
    trsd = [scores[0]]
    for i in range(len(scores)-1):
        trsd.append((scores[i]+scores[i+1])/2.)
    return numpy.array(trsd)

def get_k_subsets(dataset,k):
    # divide dataset to k equal subsets
    subsets = []
    # shuffle dataset
    inds = numpy.random.permutation(dataset.shape[0])
    split_increment = int(dataset.shape[0]/k)
    start = 0
    end = start + split_increment
    while end < len(inds) :
        subset_ids = inds[start:end]
        subsets.append(dataset[subset_ids])
        start = end
        end += split_increment
    
    subsets.append(dataset[start:end])

    return subsets

def get_k_train(i, subsets):
    # remove test set
    del subsets[i]
    # merge remaining sets
    train_set = []
    for s in subsets:
        train_set += s

    return train_set

def summary_statistics(linear, rbf):
    print('====> Linear Model: ')
    print('mean: %.2E' % Decimal(str(numpy.mean(linear))))
    print('variance: %.2E' % Decimal(str(pow(numpy.std(linear, ddof=1), 2))))

    print('====> RBF Model:')
    print('mean: %.2E' % Decimal(str(numpy.mean(rbf))))
    print('variance: %.2E' % Decimal(str(pow(numpy.std(rbf, ddof=1), 2))))

def paired_t_test(group1, group2):
    # difference in accuracies
    diff = numpy.subtract(group1, group2)
    n = len(group1)

    # compute t_score
    mean = numpy.mean(numpy.array(diff))
    sample_std = numpy.std(numpy.array(diff), ddof=1) # use ddof to compute the sample std
    t_score = round(math.sqrt(n)*mean/sample_std, 5) # round to a specific precsion for comparsion purposes
    # print('t_score: ', t_score)

    #library t-test
    lib_t, lib_p = scipy.stats.ttest_rel(group1, group2)
    # print('lib_t: ', lib_t)
    # print('lib_p: ', lib_p)

    if round(lib_t, 5) != t_score:
        print('======== Error in computing t-score!')
        return False

    df = n-1
    conf = 0.95 # confidence interval
    p = 1 - conf
    t_value = round(scipy.stats.t.ppf(conf, df), 5)
    # print('t_value: ', t_value)
    # print('p: ', p)
    
    # get significance using t-value 
    diff_t =  t_value <= abs(t_score) # reject null hypothesis => significant difference

    # get significance from lib_p value
    diff_p = abs(round(lib_p, 2)) <= round(p, 2)

    if diff_p != diff_t:
        print('========= Unmatching t and p results!')
        return False

    return diff_t

def cross_validation(k, rounds, dataset, svm_variants):
    linear = []
    rbf = []
    for r in range(rounds):
        subsets = get_k_subsets(dataset,k)
        cv_accuracy = {'linear': [], 'rbf': []}
        for i, s in enumerate(subsets):
            test_set = s
            train_subsets = subsets[:i] + subsets[i+1:]
            train_set = numpy.concatenate(train_subsets, axis=0)
            actual = test_set[:,-1].astype(int)
            for variant, params in svm_variants:
                model, svi = prepare_svm_model(train_set[:,:-1],  train_set[:,-1], **params)        
                svm_scores = svm_predict_vs(test_set[:,:-1], model)
                predicted = ((1+numpy.sign(svm_scores))/2.).astype(int)
                acc = accuracy_metric(actual, predicted)
                print('variant:', variant)
                print('total positive class', numpy.sum(predicted))
                print('final acc', acc)
                # get the accuracy metric for this cv-fold for the corresponding variant
                cv_accuracy[variant].append(acc)

        linear.append(numpy.mean(numpy.array(cv_accuracy['linear']))) # accuracy estimate from CV
        rbf.append(numpy.mean(numpy.array(cv_accuracy['rbf'])))

    return numpy.array(linear), numpy.array(rbf)

def bootstrap(rounds, dataset, svm_variants):
    cv_accuracy = {'linear': [], 'rbf': []}

    for i in range(rounds):
        r = dataset.shape[0]
        bootstrap = numpy.random.choice(r,size=r,replace=True)
        train_set = dataset[bootstrap]
        test_set = dataset
        actual = test_set[:,-1].astype(int)
        accuracy = []
        for variant, params in svm_variants:
            model, svi = prepare_svm_model(train_set[:,:-1],  train_set[:,-1], **params)        
            svm_scores = svm_predict_vs(test_set[:,:-1], model)
            predicted = ((1+numpy.sign(svm_scores))/2.).astype(int)
            acc = accuracy_metric(actual, predicted)
            accuracy.append(acc)
            cv_accuracy[variant].append(acc)
            
    return numpy.array(cv_accuracy['linear']), numpy.array(cv_accuracy['rbf'])
    


########################################################
#### DECISION TREE CLASSIFIER
########################################################

# Split a dataset based on an attribute and an attribute value
def test_split(data, dindices, vindex, value):
    mask = data[dindices, vindex] < value
    return dindices[mask], dindices[numpy.logical_not(mask)]

# Select the best split point for a dataset
def get_split(data, dindices, split_measure, weights=None, min_size=-1):
    lbls = data[:, -1].astype(int)
    ll = max(2, len(numpy.unique(lbls)))
    sc = split_measure([dindices], lbls, weights=weights)
    best = {'index': None, 'value': None, 'groups': [dindices, dindices[[]]], "score": sc}
    logs = []
    for vindex in range(data.shape[1]-1):
        splitvs = numpy.unique(data[dindices, vindex])
        for si in range(splitvs.shape[0]-1):
            #splitv = (splitvs[si]+splitvs[si+1])/2.
            splitv = splitvs[si+1]
            groups = test_split(data, dindices, vindex, splitv)
            # if any([len(g) < min_size for g in groups]):
            #     continue
            score = split_measure(groups, lbls, weights=weights)
            if VERBOSE_TREE:
                # g0 = "&\\textcolor{TolBlue}{%d} & \\textcolor{TolRed}{%d} & \\textcolor{TolYellow}{%d}&" % tuple(numpy.bincount(lbls[groups[0]], minlength=3))
                # g1 = "&\\textcolor{TolBlue}{%d} & \\textcolor{TolRed}{%d} & \\textcolor{TolYellow}{%d}&" % tuple(numpy.bincount(lbls[groups[1]], minlength=3))
                # lstr = "$v_%d \\geq %s$ & %s & %s & $%.4f$ \\\\" % (vindex+1, splitv, g0, g1, -score)
                gs = " ".join(["/".join(["%d" % d for d in numpy.bincount(lbls[g], minlength=ll)]) for g in groups])
                lstr = "v_%d \\geq %s\t%s\t%.4f" % (vindex+1, splitv, gs, -score)
                logs.append((score, lstr))
            if best is None or score < best["score"]:
                best = {'index': vindex, 'value': splitv, 'groups': groups, "score": score}

    if VERBOSE_TREE:
        if best is not None and best["index"] is not None:
            # for (sc, l) in sorted(logs):
            for (sc, l) in logs:
                print(l)
            g_counts =  (numpy.bincount(lbls[best["groups"][0]], minlength=ll), numpy.bincount(lbls[best["groups"][1]], minlength=ll))
            # g_points =  (", ".join(["%d" % ki for ki in best["groups"][0]+1]), ", ".join(["%d" % ki for ki in best["groups"][1]+1]))
            g_points =  (" ", " ")
            print("<<< MADE SPLIT $v_%d >= %s$ no/L:%d=%s {%s} yes/R:%d=%s {%s} %s" % (best["index"]+1, best["value"], len(best["groups"][0]), g_counts[0], g_points[0], len(best["groups"][1]), g_counts[1], g_points[1], best["score"]))
        else:
            print("<<< DID NOT SPLIT")
    return best

# Create a terminal node value
def to_terminal(data, dindices, weights=None):
    tst = data[dindices, -1].astype(int)
    nbdv = numpy.unique(data[:, -1]).shape[0]
    if weights is None:
        ps = numpy.bincount(tst, minlength=nbdv)/float(tst.shape[0])
    else:
        ps = numpy.bincount(tst, minlength=nbdv)*weights.astype(float)
        ps /= numpy.sum(ps)
    top = numpy.argmax(ps)
    return (top, ps[top])

# Create child splits for a node or make terminal
def split(data, node, split_measure, max_depth, min_size, depth, root=None, weights=None):
    if node is None:
        return None
    left, right = node['groups']
    both_terminals = True
    # del(node['groups'])
    # check for a no split
    if len(left) == 0 or len(right) == 0:
        node['left'] = node['right'] = to_terminal(data, numpy.hstack([left, right]), weights=weights)
        return node['right']
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(data, left, weights=weights), to_terminal(data, right, weights=weights)
        if node['left'][0] == node['right'][0]:
            return to_terminal(data, numpy.hstack([left, right]), weights=weights)
        return None
    # process left child
    if len(left) <= min_size or len(set(data[left,-1])) == 1:
        node['left'] = to_terminal(data, left, weights=weights)
    else:
        node['left'] = get_split(data, left, split_measure, weights=weights, min_size=min_size)
        if node['left'] is None:
            node['left'] = to_terminal(data, left, weights=weights)
        else:
            ret = split(data, node['left'], split_measure, max_depth, min_size, depth+1, root=root, weights=weights)
            if ret is not None:
                node['left'] = ret
            else:
                both_terminals = False
    # process right child
    if len(right) <= min_size or len(set(data[right,-1])) == 1:
        node['right'] = to_terminal(data, right, weights=weights)
    else:
        node['right'] = get_split(data, right, split_measure, weights=weights, min_size=min_size)
        if node['right'] is None:
            node['right'] = to_terminal(data, right, weights=weights)
        else:
            ret = split(data, node['right'], split_measure, max_depth, min_size, depth+1, root=root, weights=weights)
            if ret is not None:
                node['right'] = ret
            else:
                both_terminals = False
            
    if both_terminals and node['right'][0] == node['left'][0]:
        return to_terminal(data, numpy.hstack([left, right]), weights=weights)
    return None
        
def disp_tree(node, depth=0, side=" "):
    map_sides = {"l": "no ", "r": "yes"}
    sss = "%s%s |_ [v%d >= %s] score=%.3f\n" % (depth*"\t", map_sides.get(side,""), node['index']+1, node['value'], node['score'])
    for sd, gi in [("right", 1), ("left", 0)]:
        if sd in node:
            if isinstance(node[sd], dict):
                sss += disp_tree(node[sd], depth+1, sd[0])
            else:
                sss += "%s%s |_  y=%s #%d\n" % ((depth+1)*"\t", map_sides.get(sd[0],""), node[sd], len(node["groups"][gi]))
    return sss
               
# Build a decision tree
def build_tree(data, split_measure, max_depth, min_size, weights=None):
    root = get_split(data, numpy.arange(data.shape[0]), split_measure, weights=weights)
    split(data, root, split_measure, max_depth, min_size, 1, root=root, weights=weights)
    return root

# Make a prediction with a decision tree
def tree_predict_row(row, node):
    if row[node['index']] < node['value']:
        if "left" not in node:
            return (.5, .5)
        if isinstance(node['left'], dict):
            return tree_predict_row(row, node['left'])
        else:
            return node['left']
    else:
        if "right" not in node:
            return (.5, .5)
        if isinstance(node['right'], dict):
            return tree_predict_row(row, node['right'])
        else:
            return node['right']        

########################################################
#### SUPPORT VECTOR MACHINES (SVM)
########################################################
import cvxopt.solvers
MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class Kernel(object):
    
    kfunctions = {}
    kfeatures = {}
    
    def __init__(self, ktype='linear', kparams={}):
        self.ktype = 'linear'
        if ktype in self.kfunctions:
            self.ktype = ktype
        else:
            raise Warning("Kernel %s not implemented!" % self.ktype)
        self.kparams = kparams

    def distance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        return self.kfunctions[self.ktype](X, Y, **self.kparams)


def linear(X, Y):
    return numpy.dot(X, Y.T)
Kernel.kfunctions['linear'] = linear
def polynomial(X, Y, degrees=None, offs=None):
    if degrees is None:
        return linear(X, Y)
    if offs is None:
        return numpy.sum(numpy.dstack([numpy.dot(X, Y.T)**d for d in degrees]), axis=2)
    return numpy.sum(numpy.dstack([(numpy.dot(X, Y.T)+offs[i])**d for i,d in enumerate(degrees)]), axis=2)
Kernel.kfunctions['polynomial'] = polynomial
def RBF(X, Y, sigma):
    return numpy.vstack([numpy.exp(-numpy.sum((X-numpy.outer(numpy.ones(X.shape[0]), Y[yi,:]))** 2, axis=1) / (2. * sigma ** 2)).T for yi in range(Y.shape[0])]).T
Kernel.kfunctions['RBF'] = RBF

def compute_multipliers(X, y, c, kernel):
    n_samples, n_features = X.shape
    
    K = kernel.distance_matrix(X)
    P = cvxopt.matrix(numpy.outer(y, y) * K)
    q = cvxopt.matrix(-1 * numpy.ones(n_samples))
    if c == 0: # hard-margin
        G = cvxopt.matrix(numpy.eye(n_samples)*-1)
        h = cvxopt.matrix(numpy.zeros(n_samples))
    else:
        G = cvxopt.matrix(numpy.vstack((numpy.eye(n_samples)*-1, numpy.eye(n_samples))))       
        h = cvxopt.matrix(numpy.hstack((numpy.zeros(n_samples), numpy.ones(n_samples) * c)))
    A = cvxopt.matrix(numpy.array([y]), (1, n_samples))
    b = cvxopt.matrix(0.0)
    
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    return numpy.ravel(solution['x'])

def svm_predict_vs(data, model):
    xx = model["kernel"].distance_matrix(model["support_vectors"], data)
    yy = model["lmbds"] * model["support_vector_labels"]
    return model["bias"] + numpy.dot(xx.T, yy)

def prepare_svm_model(X, y, c, ktype="linear", kparams={}):
    ### WARNING: work with labels {-1, 1} !
    y = 2.*y-1 # turn y from {0, 1} to {-1, 1}
    kernel = Kernel(ktype, kparams)
        
    lagrange_multipliers = compute_multipliers(X, y, c, kernel)
    support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER # why not !=0
    
    model = {"kernel": kernel, "bias": 0.0,
             "lmbds": lagrange_multipliers[support_vector_indices],
             "support_vectors": X[support_vector_indices],
             "support_vector_labels": y[support_vector_indices]}
    pvs = svm_predict_vs(model["support_vectors"], model)
    #model["bias"] = -pvs[0]
    ### ... bias = -(max prediction for positive support vector + min prediction for positive support vector)/2
    model["bias"] = -(numpy.max(pvs)+numpy.min(pvs))/2.
    return model, support_vector_indices


def visu_plot_svm(train_set, test_set, model, svi=None):
    minmax = dataset_minmax(numpy.vstack([train_set, test_set]))
    i, j = (0,1)
    gs = []
    lims = []
    for gi in range(train_set.shape[1]):
        step_size = float(minmax[1, gi]-minmax[0, gi])/100
        gs.append(numpy.arange(minmax[0, gi]-step_size, minmax[1, gi]+1.5*step_size, step_size))
        lims.append([minmax[0, gi]-2*step_size, minmax[1, gi]+2*step_size])
    axe = plt.subplot()
    
    bckgc = (0, 0, 0, 0)
    color = "#888888"
    color_lgt = "#DDDDDD"
    cmap="coolwarm"
    ws = numpy.dot(model["lmbds"]*model["support_vector_labels"], model["support_vectors"])
    coeffs = numpy.hstack([ws, [model["bias"]]])
    sv_points = []

    vmin, vmax = (minmax[0,-1], minmax[1,-1])
    axe.scatter(train_set[:, j], train_set[:,i], c=train_set[:,-1], vmin=vmin, vmax=vmax, cmap=cmap, s=50, marker=".", edgecolors='face', linewidths=2, gid="data_points_lbl")
    axe.scatter(test_set[:, j], test_set[:,i], c=test_set[:,-1], vmin=vmin, vmax=vmax, cmap=cmap, s=55, marker="*", edgecolors='face', linewidths=2, zorder=2, gid="data_points_ulbl")

    
    if svi is not None:
        sv_points = numpy.where(svi)[0]

    xs = numpy.array([gs[j][0],gs[j][-1]])
    tmp = -(coeffs[i]*numpy.array([gs[i][0],gs[i][-1]])+coeffs[-1])/coeffs[j]
    xtr_x = numpy.array([numpy.maximum(tmp[0], xs[0]), numpy.minimum(tmp[-1], xs[-1])])

    x_pos = xtr_x[0]+.1*(xtr_x[1]-xtr_x[0])
    x_str = xtr_x[0]+.66*(xtr_x[1]-xtr_x[0])
    
    ys = -(coeffs[j]*xs+coeffs[-1])/coeffs[i]
    axe.plot(xs, ys, "-", color=color, linewidth=0.5, zorder=5, gid="details_svm_boundary")
        
    closest = (None, 0.)
    p0 = numpy.array([0, -coeffs[-1]/coeffs[i]])
    ff = numpy.array([1, -coeffs[j]/coeffs[i]])
    V = numpy.outer(ff, ff)/numpy.dot(ff, ff)
    offs = numpy.dot((numpy.eye(V.shape[0]) - V), p0)

    mrgs = [1.]
    for tii, ti in enumerate(sv_points):
        proj = numpy.dot(V, train_set[ti,[j, i]]) + offs
        axe.plot([train_set[ti,j], proj[0]], [train_set[ti,i], proj[1]], color=color_lgt, linewidth=0.25, zorder=0, gid=("details_svm_sv%d" % tii))
            
    #### plot margin
    for mrg in mrgs:
         yos = -(coeffs[j]*xs+coeffs[-1]-mrg)/coeffs[i]
         axe.plot(xs, yos, "-", color=color_lgt, linewidth=0.5, zorder=0, gid="details_svm_marginA")
         yos = -(coeffs[j]*xs+coeffs[-1]+mrg)/coeffs[i]
         axe.plot(xs, yos, "-", color=color_lgt, linewidth=0.5, zorder=0, gid="details_svm_marginB")

         mrgpx = x_pos # xs[0]+pos*(xs[-1]-xs[0])
         mrgpy = -(coeffs[j]*mrgpx+coeffs[-1]-mrg)/coeffs[i]
         mrgp = numpy.array([mrgpx, mrgpy])
         proj = numpy.dot(V, mrgp) + offs
         mrgv = numpy.sqrt(numpy.sum((mrgp-proj)**2))
         print("m=%.3f" % mrgv)
         axe.arrow(proj[0], proj[1], (proj[0]-mrgpx), (proj[1]-mrgpy), length_includes_head=True, color=color, linewidth=0.25, gid="details_svm_mwidthA")
         axe.arrow(proj[0], proj[1], -(proj[0]-mrgpx), -(proj[1]-mrgpy), length_includes_head=True, color=color, linewidth=0.25, gid="details_svm_mwidthB")

         
         axe.annotate("m=%.3f" % mrgv, (proj[0], proj[1]), (0, 15), textcoords='offset points', color=color, backgroundcolor=bckgc, zorder=10, gid="details_svm_mwidth-ann")
    axe.set_xlim(lims[j][0], lims[j][-1])
    axe.set_ylim(lims[i][0], lims[i][-1])
    plt.show()
