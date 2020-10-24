#!/usr/bin/env python
# coding: utf-8

import ipynb.fs.full.processing as processing
import ipynb.fs.full.training as training
import ipynb.fs.full.storage as storage
import ipynb.fs.full.misc as misc
import ipynb.fs.full.splitting as splitting
import ipynb.fs.full.features as features
import ipynb.fs.full.ensemble as ensemble
import os
import json
from sklearn.metrics import confusion_matrix

def testing(config):

    config
    dataframe = processing.create_dataframe(config)

    regression_dataset = features.add(dataframe, config['features'])

    primary_dataset = splitting.general(
        regression_dataset,
        config['splitting']['train_split']
    )

    regression_ensemble, regression_table = ensemble.regression(primary_dataset, config)

    decision_machine = misc.decision_machine()

    regression_labels = decision_machine.calibrate(
        regression_table,
        config['classification_ensemble']['decision']
    )

    labeled_regression_table = misc.replace_labels(
        regression_table,
        regression_labels
    )

    classifier_ensemble, classifier_table = ensemble.classifier(labeled_regression_table, config)

    classifier_matrixes = {}
    matrix_labels = labeled_regression_table['label'].to_numpy()

    for column in classifier_table.columns:
        
        # MODEL PREDICTIONS
        predictions = classifier_table[column].to_numpy()
        
        # CREATE A CONFUSION MATRIX
        matrix = confusion_matrix(
            matrix_labels,
            predictions,
            labels=[0, 1, 2]
        )
        
        # PUSH IT TO THE CONTAINER
        classifier_matrixes[column] = {
            'training': {
                'graph': 'matrix',
                'data': matrix.tolist()
            }
        }

    test_predictions = regression_ensemble.predict(primary_dataset['test'])

    test_results = test_predictions.copy()
    test_results['label'] = primary_dataset['test']['labels'][-len(test_results):]

    mash_dataset = {
        'features': test_predictions.to_numpy(),
        'labels': []
    }

    cls_dataset = classifier_ensemble.predict(mash_dataset)

    matrix_labels = decision_machine.convert(test_results)

    for column in cls_dataset.columns:
        
        # MODEL PREDICTIONS
        predictions = cls_dataset[column].to_numpy()
        
        # CREATE A CONFUSION MATRIX
        matrix = confusion_matrix(
            matrix_labels,
            predictions,
            labels=[0, 1, 2]
        )
        
        # PUSH IT TO THE MATRIX CONTAINER
        classifier_matrixes[column]['validation'] = {
            'graph': 'matrix',
            'data': matrix.tolist()
        }

    regression_fitting = {}

    for blob in regression_ensemble.models:
        collection = {}
        for index, model in enumerate(blob):
            
            # DEFAULT BAR TYPE
            bar_type = 'line'
            
            # IF THE SCORE HAS A R2 PROPERTY
            if 'R2' in model.score:
                
                # CHANGE BAR TYPE
                bar_type = 'bar'
            
                # ADD THE SUB-PROPERTIES IF THEY DONT ALREADY EXIST
                if 'R2' not in collection:
                    for key in model.score.keys():
                        collection[key] = {
                        }
                        
                # LOOP IN VALUES
                for key in model.score.keys():
                    collection[key]['fold_' + str(index)] = model.score[key]
            
            # OTHERWISE, INJECT NORMALLY
            else:
                collection['fold_' + str(index)] = model.score
            
        regression_fitting[model.name] = {
            'graph': bar_type,
            'data': collection
        }

    pipeline_name = storage.save_pipeline({
        'config': config,
        'regression_ensemble': regression_ensemble,
        'classifier_ensemble': classifier_ensemble,
        'predictions': {
            'regression': {
                'training': {
                    'graph': 'line',
                    'data': json.loads(regression_table.to_json())
                },
                'validation': {
                    'graph': 'line',
                    'data': json.loads(test_results.to_json())
                }
            },
            'classifiers': classifier_matrixes
        },
        'regression_fitting': regression_fitting
    })

    return pipeline_name




