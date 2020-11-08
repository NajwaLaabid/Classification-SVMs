import resources
import numpy
import matplotlib.pyplot as plt
import math

TASK1 = False
TASK2 = False
TASK3_4 = False
TASK3_4_BASE = False
TASK5 = True

if TASK5:
  data_params = {"filename": 'creditDE.csv', "last_column_str": True}
  dataset, header, classes = resources.load_csv(**data_params)

  train_portion = .75
  n = dataset.shape[0]

  # rare class sampling
  rare_class_idx = [i for i,c in enumerate(dataset[:,-1]) if c == 1]
  rare_portion = 1.*len(rare_class_idx)/dataset.shape[0]
  size_rare = int(n*rare_portion)
  normal_class_idx = [i for i,c in enumerate(dataset[:,-1]) if c == 0 ]
  normal_portion = .75 - rare_portion
  size_normal = int(n*normal_portion)

  print('size_normal', size_normal)
  print('size_rare', size_rare)
  normal_indices = numpy.random.choice(normal_class_idx,size=size_normal,replace=False)
  rare_indices = numpy.random.choice(rare_class_idx,size=size_rare,replace=False)
  train_indices = numpy.concatenate((normal_indices,rare_indices),axis=0)
  numpy.random.shuffle(train_indices)

  train_set = dataset[train_indices]

  # test set
  test_portion = 1 - train_portion
  test_size = int(n*test_portion)
  print('test_size', test_size)
  test_indices = numpy.random.choice(test_size,size=test_size,replace=False)
  #print(test_indices)
  test_set = dataset[test_indices]

  algo_params = {"c": 0, "ktype": "RBF", "kparams": {"sigma":1.}}
  #algo_params = {"c": 2, "ktype": "linear", "kparams": {}}

  model, svi = resources.prepare_svm_model(train_set[:,:-1], train_set[:,-1], **algo_params)        
  
  svm_scores = resources.svm_predict_vs(test_set[:,:-1], model)
  predicted = (1+numpy.sign(svm_scores))/2.

  stat, cm = resources.get_CM_vals(test_set[:,-1], predicted)
  print('===== Ensemble model results')
  print(stat)
  print(cm)


if TASK3_4_BASE:
  data_params = {"filename": 'creditDE.csv', "last_column_str": True}
  dataset, header, classes = resources.load_csv(**data_params)
  #algo_params = {"c": 0, "ktype": "RBF", "kparams": {"sigma":1.}}
  algo_params = {"c": 2, "ktype": "linear", "kparams": {}}

  train_set, test_set = resources.split_dataset(dataset, .75)

  model, svi = resources.prepare_svm_model(train_set[:,:-1], train_set[:,-1], **algo_params)        
  
  svm_scores = resources.svm_predict_vs(test_set[:,:-1], model)
  predicted = (1+numpy.sign(svm_scores))/2.

  stat, cm = resources.get_CM_vals(test_set[:,-1], predicted)
  print('===== Ensemble model results')
  print(stat)
  print(cm)

if TASK3_4:
  data_params = {"filename": 'creditDE.csv', "last_column_str": True}
  dataset, header, classes = resources.load_csv(**data_params)
  #algo_params = {"c": 0, "ktype": "RBF", "kparams": {"sigma":1.}}
  algo_params = {"c": 2, "ktype": "linear", "kparams": {}}
  train_set, test_set = resources.split_dataset(dataset, .75)

  t = 0
  T = 10
  n = dataset.shape[0] 
  w = numpy.full(n,1/n) # initial sampling weights
  e = .1 # random initial error (to get in the loop)
  models = [] # stores the T models we're about to train
  model_weights = numpy.full((T,), 0.) # weights used to combine predictions of models at the end
  b = .2 # constant to reduce weight updates 

  while t<T:
    # prepare weighted training set through sampling
    indices = numpy.random.choice(n,size=n,replace=True,p=w)
    sample = dataset[indices]
    # training model on weighted data set
    model, svi = resources.prepare_svm_model(sample[:,:-1],  sample[:,-1], **algo_params)     
    # get error rate in classification of dataset   
    svm_scores = resources.svm_predict_vs(dataset[:,:-1], model)
    predicted = (1+numpy.sign(svm_scores))/2.
    acc = resources.accuracy_metric(dataset[:,-1], predicted)
    e = round(1 - acc, 2) # error is rounded to make float comparison possible
    if e >= 0.50 or e == 0.: # only take models which perform better than random
      break
    a = b*math.log((1-e)/e)/2.
    models.append(model) 
    model_weights[t] = a
    # update individual weights based on classification results
    w = [wi*math.exp(a) if dataset[i,-1] != predicted[i] else wi*math.exp(-a) for i, wi in enumerate(w)]
    w = numpy.divide(w, numpy.sum(w)) # normalize weights
    t+=1

  # get predictions of individual models
  predictions = numpy.empty((T,test_set.shape[0]))
  for i,m in enumerate(models):
    svm_scores = resources.svm_predict_vs(test_set[:,:-1], m)
    predictions[i] = numpy.sign(svm_scores)
  
  # special case of bagging: when b = 0 => model weights = 0
  if b == 0.: model_weights[model_weights == 0.] = 1.

  # weighted average of models' predictions
  ensemble_pred = (1+numpy.sign(numpy.dot(numpy.transpose(predictions),model_weights)))/2.

  # evaluate ensemble model
  stat, cm = resources.get_CM_vals(test_set[:,-1], ensemble_pred)
  print('===== Ensemble model results')
  print(stat)
  print(cm)

if TASK2:
  data_params = {"filename": 'creditDE.csv', "last_column_str": True}
  dataset, header, classes = resources.load_csv(**data_params)

  svm_variants = [("linear", {"c": 2, "ktype": "linear", "kparams": {}}),
                ("rbf", {"c": 0, "ktype": "RBF", "kparams": {"sigma": 1}})]
  
  # cross-validation
  # print("****** Cross Validation ******")
  linear_accuracy, rbf_accuracy = resources.cross_validation(k=5, rounds=10, dataset=dataset, svm_variants=svm_variants)
  # summary statistics: mean and variance for each SVM variant
  resources.summary_statistics(linear_accuracy, rbf_accuracy)
  # significance testing
  significant = resources.paired_t_test(linear_accuracy, rbf_accuracy)
  if significant:
    print('======== The difference between the models is statistically significant.')
  else:
    print('======== The difference between the models is not statistically significant.')

  # bootstrapped samples
  print("****** Bootstrapped Samples ******")
  linear_accuracy, rbf_accuracy = resources.bootstrap(rounds=50, dataset=dataset, svm_variants=svm_variants)
  # summary statistics: mean and variance for each variant
  resources.summary_statistics(linear_accuracy, rbf_accuracy)
  # significance testing
  significant = resources.paired_t_test(linear_accuracy, rbf_accuracy)
  if significant:
    print('======== The difference between the models is statistically significant.')
  else:
    print('======== The difference between the models is not statistically significant.')

if TASK1:
  data_params = {"filename": 'creditDE.csv', "last_column_str": True}
  dataset, header, classes = resources.load_csv(**data_params)
  
  train_set, test_set = resources.split_dataset(dataset, .75)
  actual = test_set[:, -1].astype(int) # cast actual data points as int to allow comparison in CM

  svm_variants = [("hard-margin", {"c": 0, "ktype": "linear", "kparams": {}}),
                ("soft-margin", {"c": 2, "ktype": "linear", "kparams": {}}),
                ("rbf", {"c": 0, "ktype": "RBF", "kparams": {"sigma": 1}})]

  # svm_variants = [("rbf", {"c": 0, "ktype": "RBF", "kparams": {"sigma": 2}})]

  axe = plt.subplot()
  legend = []
  for variant, params in svm_variants:
    model, svi = resources.prepare_svm_model(train_set[:,:-1], train_set[:,-1], **params)        
    svm_scores = resources.svm_predict_vs(test_set[:,:-1], model)
    # get roc thresholds
    scores = numpy.sort(numpy.unique(numpy.round(svm_scores, 9))) # round scores to enable comparing floating numbers
    thresholds = resources.get_thresholds(scores)
    tpr = []
    fpr = []
    # compute tpr and fpr for every threshold
    for t in thresholds:
      predicted = (0. + (svm_scores > t)).astype(int) # cast predictions as int to allow accurate CM comparisons
      stat, cm = resources.get_CM_vals(actual, predicted)
      tpr.append(round(stat['TPR'], 3))
      fpr.append(round(stat['FPR'], 3))

    axe.plot(numpy.array(fpr), numpy.array(tpr)) # plot fpr over tpr

    auc = -numpy.trapz(tpr, fpr) # compute AUC by integrating following trapeziodum rule
    lgd = variant + ' SVM, area: %.3f' % auc # prepare plot legends including SVM variant name and AUC
    
    legend.append(lgd)

  plt.legend(legend, loc='lower right')
  plt.show()

