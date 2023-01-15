"""@author: marcosesquivelgonzalezEste script es el script central. Lo utilizo para ver tambi�n el accuracy en train(con SCV) para los mejores modelosencontrados en los otros scripts, así como comparar los mejores modelos más a fondo. Una vez vistos(y que no tengan overfitting), realizola submission con el código comentado más abajo"""import osimport sysMODELS = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))SRC = os.path.abspath(os.path.join(MODELS, os.pardir))sys.path.append(SRC)import utilsimport numpy as npimport pandas as pdfrom sklearn.neighbors import KNeighborsClassifierfrom sklearn.model_selection import cross_validatefrom sklearn.preprocessing import Normalizerfrom sklearn.preprocessing import StandardScalerfrom sklearn.preprocessing import RobustScalertrain_raw = utils.load_train_KnnImp()#train_raw = utils.load_train()train_raw = utils.merge_numerical(train_raw) #PARA MERGEAR LAS COLUMNAS VISTAS EN EDAtrain_X = utils.one_hot_encode(df = train_raw.drop(["Transported", "PassengerId"], axis = 1))StScal = StandardScaler()df_array = StScal.fit_transform(train_X)train_X_StScal = pd.DataFrame(df_array,columns=train_X.columns)NormScal = Normalizer()df_array = NormScal.fit_transform(train_X)train_X_NormScal = pd.DataFrame(df_array,columns=train_X.columns)RobScal = RobustScaler()df_array = RobScal.fit_transform(train_X)train_X_RobScal = pd.DataFrame(df_array,columns=train_X.columns)train_y = train_raw.Transportedscore_cv = cross_validate(estimator = KNeighborsClassifier(58, metric = 'correlation'),                              X = train_X_RobScal, y = train_y, cv = 10, n_jobs = -1,return_train_score=True)print("Train score: %.4f %.4f"%(np.mean(score_cv["train_score"]),np.std(score_cv["train_score"])))print("Test score: %.4f %.4f"%(np.mean(score_cv["test_score"]),np.std(score_cv["test_score"])))print(score_cv["test_score"])score_cv = cross_validate(estimator = KNeighborsClassifier(n_neighbors=34, metric='braycurtis'),                              X = train_X_NormScal, y = train_y, cv = 10, n_jobs = -1,return_train_score=True)print("Train score: %.4f %.4f"%(np.mean(score_cv["train_score"]),np.std(score_cv["train_score"])))print("Test_score: %.4f %.4f"%(np.mean(score_cv["test_score"]),np.std(score_cv["test_score"])))print(score_cv["test_score"])#Predicciones en el test"""test_raw = utils.load_test_KnnImp()test_raw = utils.merge_numerical(test_raw)test = utils.one_hot_encode(df = test_raw.drop(["PassengerId"], axis = 1))test_scaled_array = NormScal.transform(test)test_scaled = pd.DataFrame(test_scaled_array,columns=test.columns)knn = KNeighborsClassifier(n_neighbors = 47, p = 2).fit(train_X_NormScal,train_y)pred_labels = knn.predict(X = test_scaled)predicted_labels = utils.encode_labels(pred_labels)utils.generate_submission(labels = predicted_labels, method = "knn", notes = "RobScal_k_34_braycurtis_merged_knnImp43")"""