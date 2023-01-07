import osimport sysMODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))SRC = os.path.abspath(os.path.join(MODELS, os.pardir))sys.path.append(SRC)import utilsimport numpy as npimport pandas as pdfrom sklearn.neighbors import KNeighborsClassifierfrom sklearn.model_selection import cross_validatefrom sklearn.model_selection import GridSearchCVfrom sklearn.preprocessing import RobustScalerfrom sklearn.preprocessing import OneHotEncodertrain_raw = utils.load_train()train_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))train_y = train_raw.TransportedRobScal = RobustScaler()colnames = train_X.columnsdf_array = RobScal.fit_transform(train_X)train_X_scaled = pd.DataFrame(df_array,columns=colnames)"""knn = KNeighborsClassifier()k_range = list(range(1,100))pgrid = dict(n_neighbors=k_range,p=[1,2])grid = GridSearchCV(knn, pgrid, scoring='accuracy', n_jobs = -1, cv =10, return_train_score=True)grid_search = grid.fit(train_X_scaled,train_y)print(grid_search.best_params_)"""#----------------------------BASTANTE MÁS RÁPIDO ASÍ:(?)-------------------------#-----------------------Manhattan distance----------------------------------optim_score = 0.weights = ["uniform", "distance"]distance_metric = ['manhattan','euclidean','cosine','mahalanobis']params_distance = [None for n in range(3)] +  [dict(zip(['VI'], np.cov(train_X_scaled)))]for dist, param_dist in zip(distance_metric,params_distance):    for w in weights:        for k in range(1,100):            score_cv = cross_validate(estimator = KNeighborsClassifier(n_neighbors=k,                                                                        metric = dist,                                                                       metric_params = param_dist,                                                                       weights = w),                                      X = train_X_scaled, y = train_y, cv = 10, n_jobs = -1)                        print(round(np.mean(score_cv["test_score"]),5),end=' ')                        score_test = np.mean(score_cv["test_score"])                        if score_test > optim_score:                optim_score = score_test                k_optim = k                w_optim = w                        print('\n\n')            print('Para distancia = ',dist,' y weights = ',w)            print('Mejor resultado:',optim_score,' para k=',k_optim, '\n')        optim_score = 0                                                                     #Predicciones en el test"""test_raw = utils.load_test()test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))colnames = test.columnstest_scaled_array = RobScal.transform(test)test_scaled = pd.DataFrame(test_scaled_array,columns=colnames)knn = KNeighborsClassifier(n_neighbors = k_optim ,p = p_optim).fit(train_X_scaled,train_y)pred_labels = knn.predict(X = test_scaled)predicted_labels = utils.encode_labels(pred_labels)#utils.generate_submission(labels = predicted_labels, method = "knn", notes = "RobScal_k_" + str(k_optim) + "_and_p_" + str(p_optim))"""