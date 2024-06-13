class KNNWeightingMatrixProblem(ElementwiseProblem):
    def __init__(self, X, y, k_values, n_splits=5):
        self.X = X
        self.y = y
        self.k_values = k_values
        self.n_splits = n_splits
        n_points, n_features = X.shape
        super().__init__(n_var=n_points * n_features + 1, n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        n_points, n_features = self.X.shape
        weights = x[:-1]
        k_index = int(x[-1] * (len(self.k_values) - 1))
        k = self.k_values[k_index]
        weight_matrix = weights.reshape((n_points, n_features))

        kf = KFold(n_splits=self.n_splits)
        total_f1, total_roc_auc, total_accuracy, total_precision = 0, 0, 0, 0

        for train_index, test_index in kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            weight_matrix_train = weight_matrix[train_index]
            weight_matrix_test = weight_matrix[test_index]

            y_pred = []

            for i, test_point in enumerate(X_test):
                weighted_test_point = weight_matrix_test[i] * test_point
                weighted_distances = np.array([
                    np.linalg.norm(weighted_test_point - weight_matrix_train[j] * X_train[j])
                    for j in range(len(X_train))
                ])
                nn_indices = weighted_distances.argsort()[:k]
                nn_labels = y_train[nn_indices]
                predicted_label = np.argmax(np.bincount(nn_labels))
                y_pred.append(predicted_label)

            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')

            total_f1 += f1
            total_roc_auc += roc_auc
            total_accuracy += accuracy
            total_precision += precision

        avg_f1 = total_f1 / self.n_splits
        avg_roc_auc = total_roc_auc / self.n_splits
        avg_accuracy = total_accuracy / self.n_splits
        avg_precision = total_precision / self.n_splits

        combined_score = (avg_f1 + avg_roc_auc + avg_accuracy + avg_precision) / 4

        out["F"] = -combined_score

k_values = [3, 5, 7, 9]

problem = KNNWeightingMatrixProblem(X, y, k_values)

algorithm = GA(pop_size=50, eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)

best_weights = res.X[:-1].reshape((len(X), X.shape[1]))
best_k = k_values[int(res.X[-1] * (len(k_values) - 1))]

print("Best weights matrix found:\n", best_weights)
print("Best k value found:", best_k)
print("Maximum combined score achieved:", -res.F[0])
