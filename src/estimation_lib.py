# Omid55

import numpy as np
import cvxpy as cp
from collections import defaultdict
from sklearn.model_selection import KFold
import sys
from datetime import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D, LSTM, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Concatenate, Reshape, Embedding, Dot
from tensorflow.keras.models import Model
import utils


def compute_error(
    y_train_or_validation_or_test_true,
    y_train_or_validation_or_test_predicted,
    estimation_name,
    error_type_str):
    """Computes the error."""
    err = 0
    for index in range(len(y_train_or_validation_or_test_true)):
        groundtruth = y_train_or_validation_or_test_true[index][estimation_name]
        predicted = y_train_or_validation_or_test_predicted[index]
        # TODO(@Omid): CHECK WHETHER THIS IS JUST AN INT THEN FIND THE MOST INFLUENTIAL PERSON FROM THE ESTIAMTED MATRIX.
        err += utils.matrix_estimation_error(
            true_matrix=groundtruth,
            pred_matrix=predicted,
            type_str=error_type_str)
    err /= len(y_train_or_validation_or_test_true)
    return err


# Boilerplate model functions.

def model_builder(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        estimation_name='influence_matrix',
        error_type_str='normalized_frob_norm',
        tune_hyperparameters_by_validation=True,
        with_replication=True,
        lambdas = [0, 0.1, 1, 10, 100, 1000],
        model_func='average',
        params={'with_constraints': True, 'n_splits': 3}):
    
    # For the baseline models.
    if model_func == 'average':
        mats = []
        for i in range(len(y_train)):
            mats.append(y_train[i][estimation_name])
        y_baseline_predicted = [np.matrix(np.mean(mats, axis=0)) for _ in range(len(y_train))]
    elif model_func == 'uniform':
        y_baseline_predicted = [np.matrix(np.ones((4, 4)) * 0.25) for _ in range(len(y_train))]
    elif model_func == 'random':
        y_baseline_predicted = [np.matrix(
            utils.make_matrix_row_stochastic(
                np.random.rand(4, 4))) for _ in range(len(y_train))]
    if model_func in ['average', 'uniform', 'random']:
        train_error = compute_error(
            y_train, y_baseline_predicted, estimation_name=estimation_name, error_type_str=error_type_str)
        test_error = compute_error(
            y_test, y_baseline_predicted, estimation_name=estimation_name, error_type_str=error_type_str)
        return train_error, test_error, None

    # For the proposed models.
    validation_errors = defaultdict(lambda: 0)
    if tune_hyperparameters_by_validation:
        print('{}-fold validation ...'.format(params['n_splits']))
        kf = KFold(n_splits=params['n_splits'])
        for train_index, validation_index in kf.split(X_train):
            X_train_subset, X_validation = X_train[train_index], X_train[validation_index]
            y_train_subset, y_validation = y_train[train_index], y_train[validation_index]
            if with_replication:
                print('Replicating ...')
                X_train_subset, y_train_subset = utils.replicate_matrices_in_train_dataset_with_reordering(
                    X_train_subset, y_train_subset)
                X_train_subset = np.array(X_train_subset)
                y_train_subset = np.array(y_train_subset)
            print('Shapes of train: {}, validation: {}, test: {}.'.format(
                X_train_subset.shape, X_validation.shape, X_test.shape))
            for lambdaa in lambdas:
                validation_errors[lambdaa] += model_func(
                    X_train=X_train_subset,
                    y_train=y_train_subset,
                    X_validation_or_test=X_validation,
                    y_validation_or_test=y_validation,
                    feature_names=feature_names,
                    estimation_name=estimation_name,
                    lambdaa=lambdaa,
                    error_type_str=error_type_str,
                    params=params)[1]
        best_lambda = min(validation_errors, key=validation_errors.get)
    else:
        best_lambda = 0.1
    print('Training with the best lambda: {} on entire training set...'.format(best_lambda))
    if with_replication:
        print('Replicating ...')
        X_train, y_train = utils.replicate_matrices_in_train_dataset_with_reordering(
            X_train, y_train)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
    train_error, test_error = model_func(
        X_train=X_train,
        y_train=y_train,
        X_validation_or_test=X_test,
        y_validation_or_test=y_test,
        feature_names=feature_names,
        estimation_name=estimation_name,
        lambdaa=best_lambda,
        error_type_str=error_type_str,
        params=params)
    return train_error, test_error, validation_errors


# Specific model functions.
def convex_optimization_model_func(
        X_train,
        y_train,
        X_validation_or_test,
        y_validation_or_test,
        feature_names,
        estimation_name,
        lambdaa,
        error_type_str,
        params={'with_constraints': True}):

    def predict(data_element, feature_names, B, Ws, is_solved=False):
        """Defines the prediction function."""
        predicted = 0
        if is_solved:
            predicted += B.value
        else:
            predicted += B
        for feature_name in feature_names:
            if is_solved:
                W_for_this_feature = Ws[feature_name].value
            else:
                W_for_this_feature = Ws[feature_name]
            if len(data_element[feature_name].shape) == 1:
                # If it was a vector, it makes a matrix with it.
                p = data_element[feature_name]
                data_element_matrix = np.row_stack([p, p, p, p])
            else:
                # Unless it is already a matrix.
                data_element_matrix = data_element[feature_name]
            predicted += (data_element_matrix * W_for_this_feature)
        return predicted

    # def predict(data_element, feature_names, B, Ws, is_solved=False):
    #     """Defines the prediction function."""
    #     predicted = 0
    #     if is_solved:
    #         predicted += B.value
    #     else:
    #         predicted += B
    #     for feature_name in feature_names:
    #         if is_solved:
    #             W_for_this_feature = Ws[feature_name].value
    #         else:
    #             W_for_this_feature = Ws[feature_name]
    #         predicted += (data_element[feature_name].T * W_for_this_feature)
    #     return predicted
    
    def predict_all_after_solving(X_train_or_validation_or_test, B, Ws, feature_names):
        """Predicts for all data points in the given set."""
        return [
            predict(
                data_element=data_element,
                feature_names=feature_names,
                B=B,
                Ws=Ws,
                is_solved=True) 
            for data_element in X_train_or_validation_or_test]

    # Creating variables.
    Ws = {}
    for feature_name in feature_names:
        if len(X_train[0][feature_name].shape) == 1:
            # If it was a vector, it makes a matrix with it.
            Ws[feature_name] = cp.Variable(
                len(X_train[0][feature_name]), len(X_train[0][feature_name]))
            # Ws[feature_name] = cp.Variable(
            #     len(X_train[0][feature_name]), 1)
        else:
            # Unless it is already a matrix.
            Ws[feature_name] = cp.Variable(
                X_train[0][feature_name].shape[1], X_train[0][feature_name].shape[0])
    B = cp.Variable(4, 4)

    # Computing loss.
    print('Computing convex loss on {} training data samples.'.format(
        len(X_train)))
    constraints = []
    losses = 0
    for index in range(len(X_train)):
        element = X_train[index]
        estimation_groundtruth = y_train[index][estimation_name]

        # Defining the estimation function.
        estimation_predicted = predict(
            data_element=element, feature_names=feature_names, B=B, Ws=Ws, is_solved=False)

        # Defining the loss function.
        loss = cp.sum_squares(estimation_predicted - estimation_groundtruth)

        losses += loss
        if params['with_constraints']:
            constraints += [estimation_predicted >= 0]
            constraints += [cp.sum_entries(estimation_predicted, axis=1) == 1]

    # Computing regularization.
    regluarization = cp.norm1(B)
    for feature_name in feature_names:
        regluarization += cp.norm1(Ws[feature_name])

    # Solving the convex problem.
    objective = cp.Minimize(losses + lambdaa * regluarization)
    prob = cp.Problem(objective, constraints)
    try:
        result = prob.solve(solver=cp.MOSEK)
    except cp.SolverError as e:
        print('Problem was not solved for lambda: {}'.format(lambdaa))
        return sys.maxsize, sys.maxsize
    print('The status of solution was: {} and the result was: {}'.format(prob.status, result))

    # Predicting and computing trian error.
    y_train_predicted = predict_all_after_solving(
        X_train_or_validation_or_test=X_train, B=B, Ws=Ws, feature_names=feature_names)
    train_error = compute_error(
        y_train_or_validation_or_test_true=y_train,
        y_train_or_validation_or_test_predicted=y_train_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)
    
    # Predicting and computing validation or test error.
    y_validation_or_test_predicted = predict_all_after_solving(
        X_train_or_validation_or_test=X_validation_or_test, B=B, Ws=Ws, feature_names=feature_names)
    validation_or_test_error = compute_error(
        y_train_or_validation_or_test_true=y_validation_or_test,
        y_train_or_validation_or_test_predicted=y_validation_or_test_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)
    # utils.save_it({'Ws': Ws, 'B': B}, 'cvxvars_{}.pkl'.format(datetime.now()))
    return train_error, validation_or_test_error


def concatinated_deep_neural_network_model_func(
        X_train,
        y_train,
        X_validation_or_test,
        y_validation_or_test,
        feature_names,
        estimation_name,
        lambdaa,
        error_type_str,
        params={'n_epochs': 10, 'batch_size': 32}):

    flatten_X_train = []
    flatten_y_train = []
    for i in range(len(X_train)):
        features = X_train[i]
        label = y_train[i][estimation_name]            
        flatten_X_train.append(np.hstack(
            [np.array(features[feature_name].flatten())[0] for feature_name in feature_names]))
        flatten_y_train.append(np.array(label.flatten())[0])
    flatten_X_train = np.array(flatten_X_train)
    flatten_y_train = np.array(flatten_y_train)

    flatten_X_validation_or_test = []
    flatten_y_validation_or_test = []
    for i in range(len(X_validation_or_test)):
        features = X_validation_or_test[i]
        label = y_validation_or_test[i][estimation_name]
        flatten_X_validation_or_test.append(np.hstack(
            [np.array(features[feature_name].flatten())[0] for feature_name in feature_names]))
        flatten_y_validation_or_test.append(np.array(label.flatten())[0])
    flatten_X_validation_or_test = np.array(flatten_X_validation_or_test)
    flatten_y_validation_or_test = np.array(flatten_y_validation_or_test)
                              
    _, input_size = flatten_X_train.shape
    print('Input size for the neural network was: {}'.format(input_size))

    model = Sequential([
        Dense(
            units=32,
            kernel_initializer='he_normal',
            activation='elu',
            input_shape=(input_size,),
            kernel_regularizer=regularizers.l1(lambdaa),
            activity_regularizer=regularizers.l1(lambdaa)),
#         Dropout(0.5),
        Dense(
            units=64,
            kernel_initializer='he_normal',
            activation='elu',
            kernel_regularizer=regularizers.l1(lambdaa),
            activity_regularizer=regularizers.l1(lambdaa)),
#         Dropout(0.5),
        Dense(
            units=32,
            kernel_initializer='he_normal',
            activation='elu',
            kernel_regularizer=regularizers.l1(lambdaa),
            activity_regularizer=regularizers.l1(lambdaa)),
#         Dropout(0.5),
        Dense(16, kernel_initializer='glorot_uniform', activation='sigmoid')])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(flatten_X_train, flatten_y_train, epochs=params['n_epochs'], batch_size=params['batch_size'])

    # Predicting and computing train error.
    y_train_predicted = [utils.make_matrix_row_stochastic(
        np.matrix(np.reshape(element, (4, 4)))) for element in model.predict(flatten_X_train)]
    train_error = compute_error(
        y_train_or_validation_or_test_true=y_train,
        y_train_or_validation_or_test_predicted=y_train_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)

    # Predicting and computing train error.
    y_validation_or_test_predicted = [utils.make_matrix_row_stochastic(
        np.matrix(np.reshape(element, (4, 4)))) for element in model.predict(flatten_X_validation_or_test)]
    validation_or_test_error = compute_error(
        y_train_or_validation_or_test_true=y_validation_or_test,
        y_train_or_validation_or_test_predicted=y_validation_or_test_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)

    return train_error, validation_or_test_error


def last_model_func(
        X_train,
        y_train,
        X_validation_or_test,
        y_validation_or_test,
        feature_names=[],
        estimation_name='influence_matrix',
        lambdaa=[],
        error_type_str='mse',
        params={}):
    """Baseline model that always predicts no change in the influence matrix."""
    y_validation_or_test_predicted = [
        item['previous_influence_matrix']
        for item in X_validation_or_test]
    validation_or_test_error = compute_error(
        y_train_or_validation_or_test_true=y_validation_or_test,
        y_train_or_validation_or_test_predicted=y_validation_or_test_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)
    return -1, validation_or_test_error


def sbt_model_func(
        X_train,
        y_train,
        X_validation_or_test,
        y_validation_or_test,
        feature_names=[],
        estimation_name='influence_matrix',
        lambdaa=[],
        error_type_str='mse',
        params={'mode': 1}):
    """Structural Balance Theory model inspired (similar to Kulakowski)."""
    if 'mode' in params:
        mode = params['mode']
    else:
        mode = 1
    y_validation_or_test_predicted = []
    for item in X_validation_or_test:
        influence_matrix = item['previous_influence_matrix'] 
        n, m = influence_matrix.shape
        if n != m:
            raise ValueError('The matrix was not squared.')
        next_influence_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    ks = list(set.difference(set(range(n)), [i, j]))
                    wij = 0
                    for k in ks:
                        wij += influence_matrix[i, k] * influence_matrix[k, j] 
    #                 wij /= (n - 2)
                    next_influence_matrix[i, j] = wij
        if mode == 1:
            # Fill the diagonal with previous influence matrix and normalize to become row-stochastic.
            np.fill_diagonal(next_influence_matrix, np.diag(influence_matrix))
            next_influence_matrix = utils.make_matrix_row_stochastic(
                next_influence_matrix)
        elif mode == 2:
            # Fill the diagonal with 1 - sum of the current filled row.
            np.fill_diagonal(
                next_influence_matrix, 1 - np.sum(
                    next_influence_matrix, axis=1))
        else:
            raise ValueError(
                'The input mode was wrong. It was {}'.format(mode))
        y_validation_or_test_predicted.append(next_influence_matrix)
    validation_or_test_error = compute_error(
        y_train_or_validation_or_test_true=y_validation_or_test,
        y_train_or_validation_or_test_predicted=y_validation_or_test_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)
    return -1, validation_or_test_error


def mei_inspired_model_func(
        X_train,
        y_train,
        X_validation_or_test,
        y_validation_or_test,
        feature_names=[],
        estimation_name='influence_matrix',
        lambdaa=[],
        error_type_str='mse',
        params={}):
    """Model inspired by the Mei et al. 2017 study."""
    y_validation_or_test_predicted = []
    for item in X_validation_or_test:
        influence_matrix = item['previous_influence_matrix'] 
        n, m = influence_matrix.shape
        if n != m:
            raise ValueError('The matrix was not squared.')
        perfs = item['individual_performance']
        p_minus_Mp = np.diag(perfs - np.mean(perfs))
        A_dot = (p_minus_Mp * np.diag(np.diag(influence_matrix)) * (
            np.eye(n) - influence_matrix)) / np.sum(perfs)
        y_validation_or_test_predicted.append(influence_matrix + A_dot)
    validation_or_test_error = compute_error(
        y_train_or_validation_or_test_true=y_validation_or_test,
        y_train_or_validation_or_test_predicted=y_validation_or_test_predicted,
        estimation_name=estimation_name,
        error_type_str=error_type_str)
    return -1, validation_or_test_error
    