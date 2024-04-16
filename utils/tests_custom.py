import onnx.onnx_ml_pb2
import pytest
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
import onnx
from typing import Union
import onnxruntime


class Tests:
  def __init__(self):
    self.passed = 0

  def testModel(self, pipeline, x_training, y_training, x_testing, y_testing, accuracy=True, roc_auc=True, f1=True, precision=True, recall=True, conf_matrix=True, cross_validate=False, roc_curves=True):
    '''Test the model on lots of different metrics. Default test.'''
    print("Training the model")
    pipeline.fit(x_training, y_training)
    y_pred = pipeline.predict(x_testing)
    if accuracy:
      print(f'Accuracy:  {accuracy_score(y_testing, y_pred)}')
    if roc_auc:
      print(f'roc_auc:   {roc_auc_score(y_testing, pipeline.predict_proba(x_testing)[:, 1])}')
    if f1:
      print(f'f1:        {f1_score(y_testing, y_pred)}')
    if precision:
      print(f'precision: {precision_score(y_testing, y_pred)}')
    if recall:
      print(f'recall:    {recall_score(y_testing, y_pred)}')
    if conf_matrix:
      print('confusion matrix:')
      print(confusion_matrix(y_testing, y_pred))
    if cross_validate:
      print("Cross-Validation Scores: ", cross_val_score(pipeline, x_training, y_training, cv=5))
    if(roc_curves):
      y_scores = pipeline.predict_proba(x_testing)[:, 1]
      fpr, tpr, thresholds = roc_curve(y_testing, y_scores)
      roc_auc = auc(fpr, tpr)

      plt.figure()
      plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend(loc="lower right")
      plt.show()

  def testManWoman(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame):
    '''Changes the singular variable "persoon_geslacht_vrouw" to all men, all women, random and gender flipped and evaluates accuracy.'''
    # Original
    y_orig = pipeline.predict(X_test)

    # Flip gender labels
    X_test_flip = X_test.copy()
    X_test_flip['persoon_geslacht_vrouw'] = round(1.0 - X_test_flip['persoon_geslacht_vrouw'])

    y_flip = pipeline.predict(X_test_flip)

    # All male
    X_test_male = X_test.copy()
    X_test_male['persoon_geslacht_vrouw'] = 0.0

    y_male = pipeline.predict(X_test_male)

    # All female
    X_test_female = X_test.copy()
    X_test_female['persoon_geslacht_vrouw'] = 1.0

    y_female = pipeline.predict(X_test_female)

    # All random
    X_test_random = X_test.copy()
    X_test_random['persoon_geslacht_vrouw'] = np.random.randint(0, 2, len(X_test_random['persoon_geslacht_vrouw']))

    y_random = pipeline.predict(X_test_random)

    print('Original Accuracy', accuracy_score(y_test, y_orig))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_orig-y_test))
    print('FLipped Accuracy', accuracy_score(y_test, y_flip))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_test-y_flip))
    print('All Male Accuracy', accuracy_score(y_test, y_male))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_test-y_male))
    print('All Female Accuracy', accuracy_score(y_test, y_female))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_test-y_female))
    print('All Random Accuracy', accuracy_score(y_test, y_random))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_test-y_random))

    return y_orig, y_flip, y_male, y_female, y_random
  
  def testGenderOnnx(self, session: onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, X_test: pd.DataFrame, y_test: pd.DataFrame, which: Union[str, list[str]] = 'all', verbose: bool = False) -> list[dict]:
    '''Perform simple gender change tests. Includes swapping genders, all male, all female and random.
       Argument 'which' needs to be a subset of ['all', 's', 'm', 'f', 'r']
       Returns a list of dicts, with the dict consisting of the test id as key, and the accuracy and euclidian distance to ground truth as values.
    '''
    assert type(which) == str or type(which) == list[str]
    if type(which) == str: which = [which]
    return_values = {}

    print('RUNNING GENDER MUTATION TESTS:')

    # Original
    if type(session) == Pipeline: y_orig = [session.predict(X_test)]
    else: y_orig = session.run(None, {'X': X_test.values.astype(np.float32)})
    if verbose: print(y_orig[0])
    acc_orig = accuracy_score(y_test, y_orig[0])
    zeros_fraction = np.mean(y_orig[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    print('Original Accuracy: ', acc_orig, 'mean risk score:', np.mean(y_orig[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_orig[0]-y_test))
    return_values['orig'] = [acc_orig, np.linalg.norm(y_orig[0]-y_test)]

    # All male
    for ele in which:
      if ele in ['all', 'm']:
        X_test_male = X_test.copy()
        X_test_male['persoon_geslacht_vrouw'] = 0.0
        if type(session) == Pipeline: y_male = [session.predict(X_test_male)]
        else: y_male = session.run(None, {'X': X_test_male.values.astype(np.float32)})
        if verbose: print(y_male[0])
        zeros_fraction = np.mean(y_male[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc_male = accuracy_score(y_test, y_male[0])
        print('All Male Accuracy: ', acc_male, 'mean risk score:', np.mean(y_male[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y_male[0]-y_test))
        return_values['m'] = [acc_male, np.linalg.norm(y_male[0]-y_test)]
        break
    
    # All female
    for ele in which:
      if ele in ['all', 'f']:
        X_test_fem = X_test.copy()
        X_test_fem['persoon_geslacht_vrouw'] = 1.0
        if type(session) == Pipeline: y_fem = [session.predict(X_test_fem)]
        else: y_fem = session.run(None, {'X': X_test_fem.values.astype(np.float32)}) 
        if verbose: print(y_fem[0])
        zeros_fraction = np.mean(y_fem[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc_fem = accuracy_score(y_test, y_fem[0])
        print('All Female Accuracy: ', acc_fem, 'mean risk score:', np.mean(y_fem[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y_fem[0]-y_test))
        return_values['f'] = [acc_fem, np.linalg.norm(y_fem[0]-y_test)]
        break

    # Swapping genders
    for ele in which:
      if ele in ['all', 's']:
        X_test_swap = X_test.copy()
        X_test_swap['persoon_geslacht_vrouw'] = round(1.0 - X_test_swap['persoon_geslacht_vrouw'])
        if type(session) == Pipeline: y_swap = [session.predict(X_test_swap)]
        else: y_swap = session.run(None, {'X': X_test_swap.values.astype(np.float32)}) 
        zeros_fraction = np.mean(y_swap[0] == 0)
        if verbose: print(y_swap[0])
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc_swap = accuracy_score(y_test, y_swap[0])
        print('Swapped Genders Accuracy: ', acc_swap, 'mean risk score:', np.mean(y_swap[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y_swap[0]-y_test))
        return_values['s'] = [acc_swap, np.linalg.norm(y_swap[0]-y_test)]
        break

    # Random genders
    for ele in which:
      if ele in ['all', 'r']:
        X_test_rand = X_test.copy()
        X_test_rand['persoon_geslacht_vrouw'] = np.random.randint(0, 2, len(X_test_rand['persoon_geslacht_vrouw']))
        if type(session) == Pipeline: y_rand = [session.predict(X_test_rand)]
        else: y_rand = session.run(None, {'X': X_test_rand.values.astype(np.float32)}) 
        if verbose: print(y_rand[0])
        zeros_fraction = np.mean(y_rand[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc_rand = accuracy_score(y_test, y_rand[0])
        print('Random Genders Accuracy: ', acc_rand, 'mean risk score:', np.mean(y_rand[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y_rand[0]-y_test))
        return_values['r'] = [acc_rand, np.linalg.norm(y_rand[0]-y_test)]
        break

    return return_values

  def testAgeOnnx(self, session: Union[onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, Pipeline], X_t: pd.DataFrame, y_test: pd.DataFrame, which: Union[str, int, list[str], list[int]] = 'all', verbose: bool = False) -> list[dict]:
    '''Perform age change tests. Includes setting ages to all 25, all 45, all 65, all random (uniform), all random (normal), and checking for lowest accuracy by iterating through all ages.
       Argument 'which' needs to be a subset of ['all', '25', '45', '65', 'u', 'n', 't']
       Returns a list of dicts, with the dict consisting of the test id as key, and the accuracy and euclidian distance to ground truth as values.
    '''
    assert type(which) == str or type(which) == list[str] or type(which) == int or type(which) == list[int]
    if type(which) == str or type(which) == int: which = [which]
    return_values = {}

    print('RUNNING AGE MUTATION TESTS:')

    # Original
    if type(session) == Pipeline: y_orig = [session.predict(X_t)]
    else: y_orig = session.run(None, {'X': X_t.values.astype(np.float32)}) 
    if verbose: print(y_orig[0])
    acc_orig = accuracy_score(y_test, y_orig[0])
    zeros_fraction = np.mean(y_orig[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    print('Original Accuracy: ', acc_orig, 'mean risk score:', np.mean(y_orig[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y_orig[0]-y_test))
    return_values['orig'] = [acc_orig, np.linalg.norm(y_orig[0]-y_test)]

    # All 25
    for ele in which:
      if ele in ['all', 25, '25']:
        X_test = X_t.copy()
        X_test['persoon_leeftijd_bij_onderzoek'] = 25
        if type(session) == Pipeline: y = [session.predict(X_test)]
        else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
        if verbose: print(y[0])
        zeros_fraction = np.mean(y[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc = accuracy_score(y_test, y[0])
        print('All 25 Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
        return_values['25'] = [acc, np.linalg.norm(y[0]-y_test)]
        break
    
    # All 45
    for ele in which:
      if ele in ['all', 45, '45']:
        X_test = X_t.copy()
        X_test['persoon_leeftijd_bij_onderzoek'] = 45
        if type(session) == Pipeline: y = [session.predict(X_test)]
        else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
        if verbose: print(y[0])
        zeros_fraction = np.mean(y[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc = accuracy_score(y_test, y[0])
        print('All 45 Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
        return_values['45'] = [acc, np.linalg.norm(y[0]-y_test)]
        break

    # All 65
    for ele in which:
      if ele in ['all', 65, '65']:
        X_test = X_t.copy()
        X_test['persoon_leeftijd_bij_onderzoek'] = 65
        if type(session) == Pipeline: y = [session.predict(X_test)]
        else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
        if verbose: print(y[0])
        zeros_fraction = np.mean(y[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc = accuracy_score(y_test, y[0])
        print('All 65 Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
        return_values['65'] = [acc, np.linalg.norm(y[0]-y_test)]
        break 

    # Uniform random ages
    for ele in which:
      if ele in ['all', 'u']:
        X_test = X_t.copy()
        X_test['persoon_leeftijd_bij_onderzoek'] = np.random.randint(min(X_test['persoon_leeftijd_bij_onderzoek']), max(X_test['persoon_leeftijd_bij_onderzoek']), len(X_test['persoon_leeftijd_bij_onderzoek']))
        if type(session) == Pipeline: y = [session.predict(X_test)]
        else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
        if verbose: print(y[0])
        zeros_fraction = np.mean(y[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc = accuracy_score(y_test, y[0])
        print('All Random (Uniform) Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
        return_values['u'] = [acc, np.linalg.norm(y[0]-y_test)]
        break

    # Random ages drawn from normal distribution
    for ele in which:
      if ele in ['all', 'n']:
        X_test = X_t.copy()
        # Random samples according to mean and std of real distribution
        X_test['persoon_leeftijd_bij_onderzoek'] = np.random.normal(loc=49.40332147093713, scale=9.822235974220519, size=len(X_test['persoon_leeftijd_bij_onderzoek']))
        if type(session) == Pipeline: y = [session.predict(X_test)]
        else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
        if verbose: print(y[0])
        zeros_fraction = np.mean(y[0] == 0)
        if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
        acc = accuracy_score(y_test, y[0])
        print('All Random (Normal) Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
        print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
        return_values['n'] = [acc, np.linalg.norm(y[0]-y_test)]
        break

    # Check all ages, keep track of 5 lowest accuracies
    lowest = {}
    for age in range(int(min(X_test['persoon_leeftijd_bij_onderzoek'])), int(max(X_test['persoon_leeftijd_bij_onderzoek']))):
        X_test = X_t.copy()
        X_test['persoon_leeftijd_bij_onderzoek'] = age
        if type(session) == Pipeline: y = [session.predict(X_test)]
        else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
        acc = accuracy_score(y_test, y[0])
        # Keep track of 5 lowest accuracies
        lower = 0
        if len(lowest) < 5:
            lowest[age] = acc
        else:
            max_current = list(lowest.keys())[np.argmax(lowest.values())]
            if acc < lowest[max_current]:
                lowest.pop(max_current)
                lowest[age] = acc
    print('Lowest accuracies found at {age, accuracy}:', lowest)
    return_values['t'] = lowest

    return return_values

  def testImmigrantsOnnx(self, session: onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, X_t: pd.DataFrame, y_test: pd.DataFrame, which: Union[str, list[str]] = 'all', verbose: bool = False) -> list[dict]:
    """Perform tests on possible feature proxies for immigrants."""

    return_values = {}

    print('RUNNING IMMIGRANT PROXY MUTATION TESTS:')

    print('persoonlijke_eigenschappen_spreektaal_anders:')
    
    # All Dutch Speakers
    X_test = X_t.copy()
    X_test['persoonlijke_eigenschappen_spreektaal_anders'] = 0.0
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All Dutch Speakers Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['ad'] = [acc, np.linalg.norm(y[0]-y_test)]
    
    # All Non-Dutch Speakers
    X_test = X_t.copy()
    X_test['persoonlijke_eigenschappen_spreektaal_anders'] = 1.0
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All Non-Dutch Speakers Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['and'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Inverted
    X_test = X_t.copy()
    X_test['persoonlijke_eigenschappen_spreektaal_anders'] = round(1.0 - X_test['persoonlijke_eigenschappen_spreektaal_anders'])
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All Inverted Speakers Accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['inv'] = [acc, np.linalg.norm(y[0]-y_test)]

    print('adres_dagen_op_adres:')

    # Set all days since address change to mean
    X_test = X_t.copy()
    X_test['adres_dagen_op_adres'] = round(np.mean(X_test['adres_dagen_op_adres']))
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All set to mean accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['doa_m'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Set all days since address change to max
    X_test = X_t.copy()
    X_test['adres_dagen_op_adres'] = np.max(X_test['adres_dagen_op_adres'])
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All max days accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['doa_max'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Set all days since address change to min
    X_test = X_t.copy()
    X_test['adres_dagen_op_adres'] = np.min(X_test['adres_dagen_op_adres'])
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All min days accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['doa_min'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Draw random samples from power-law distribution
    # TODO

    print('persoonlijke_eigenschappen_dagen_sinds_taaleis:')

    # Set all to mean
    X_test = X_t.copy()
    X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'] = round(np.mean(X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis']))
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All mean days accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['dst_mean'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Set all to max
    X_test = X_t.copy()
    X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'] = np.max(X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'])
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All max days accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['dst_max'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Set all to min
    X_test = X_t.copy()
    X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'] = np.min(X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'])
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All min days accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['dst_min'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Draw random samples from power-law distribution
    # TODO

    print('typering_hist_inburgeringsbehoeftig:')

    # Set all to zero
    X_test = X_t.copy()
    X_test['typering_hist_inburgeringsbehoeftig'] = 0.0
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All zeros accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['thi_zeros'] = [acc, np.linalg.norm(y[0]-y_test)]

    # Set all to ones
    X_test = X_t.copy()
    X_test['typering_hist_inburgeringsbehoeftig'] = 1.0
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('All ones accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['thi_ones'] = [acc, np.linalg.norm(y[0]-y_test)]

    return return_values

  def testYoungImmigrantsOnnx(self, session: onnxruntime.capi.onnxruntime_inference_collection.InferenceSession, X_t: pd.DataFrame, y_test: pd.DataFrame, which: Union[str, list[str]] = 'all', verbose: bool = False) -> list[dict]:
    """Tests the combination of being young and an immigrant. Based on findings from testImmigrantsOnnx."""
    return_values = {}

    print('RUNNING YOUNG IMMIGRANT TESTS')

    # 'Best case' scenario
    X_test = X_t.copy()
    X_test['adres_dagen_op_adres'] = np.max(X_test['adres_dagen_op_adres']) # Has place to stay longterm
    X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'] = np.max(X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis']) # Long time since language test
    X_test['persoon_leeftijd_bij_onderzoek'] = 45.0#np.max(X_test['persoon_leeftijd_bij_onderzoek'])
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('Best case accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['best'] = [acc, np.linalg.norm(y[0]-y_test)]

    # 'Worst case' scenario
    X_test = X_t.copy()
    X_test['adres_dagen_op_adres'] = np.min(X_test['adres_dagen_op_adres']) # Has place to stay longterm
    X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'] = np.min(X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis']) # Long time since language test
    X_test['persoon_leeftijd_bij_onderzoek'] = 20
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('Young Immigrant accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['young'] = [acc, np.linalg.norm(y[0]-y_test)]

    # 'Worst case' scenario
    X_test = X_t.copy()
    X_test['adres_dagen_op_adres'] = np.min(X_test['adres_dagen_op_adres']) # Has place to stay longterm
    X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis'] = np.min(X_test['persoonlijke_eigenschappen_dagen_sinds_taaleis']) # Long time since language test
    X_test['persoon_leeftijd_bij_onderzoek'] = 65
    if type(session) == Pipeline: y = [session.predict(X_test)]
    else: y = session.run(None, {'X': X_test.values.astype(np.float32)}) 
    if verbose: print(y[0])
    zeros_fraction = np.mean(y[0] == 0)
    if zeros_fraction > 0.98: print('WARNING: Prediction was almost all zeros')
    acc = accuracy_score(y_test, y[0])
    print('Worst case accuracy: ', acc, 'mean risk score:', np.mean(y[0]))
    print('Euclidian Distance to Ground Truth', np.linalg.norm(y[0]-y_test))
    return_values['worst'] = [acc, np.linalg.norm(y[0]-y_test)]

    return return_values

  # Generates a completely noisy dataset in the same shape as the original. There should be no correlations to learn from here
  def generate_noise(self, original_df: pd.DataFrame):
    noise_df = pd.DataFrame()

    for column in original_df.columns:
        unique_values = original_df[column].unique()

        # If the column has only two unique values (boolean), generate random boolean noise
        if len(unique_values) == 2 and all(isinstance(v, bool) for v in unique_values):
            noise_df[column] = np.random.choice([True, False], size=len(original_df))
        # If the column has discrete values, generate random noise from those values
        elif len(unique_values) <= 10:  # Adjust threshold as needed
            noise_df[column] = np.random.choice(unique_values, size=len(original_df))
        # Otherwise, generate random noise from a uniform distribution
        else:
            min_value, max_value = original_df[column].min(), original_df[column].max()
            noise_df[column] = np.random.randint(min_value, max_value + 1, size=len(original_df))

    return noise_df.copy()

  def testNoise(self, pipeline: Pipeline, dataFrame: pd.DataFrame):
    '''Generates a dataset consisting of pure noise.'''
    # Suppress specific performance warning
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    X_noise = self.generate_noise(original_df=dataFrame).drop(['checked'], axis=1)
    X_noise = X_noise
    y_noise = dataFrame['checked']

    X_train_noise, X_test_noise, y_train_noise, y_test_noise = train_test_split(X_noise, y_noise, test_size=0.25, random_state=0)
    self.testModel(pipeline, X_train_noise, y_train_noise, X_test_noise, y_test_noise)

  def testAlmostZero(self, pipeline: Pipeline, data: pd.DataFrame = None, X_test: pd.DataFrame = None, y_test: pd.DataFrame = None) -> bool:
    '''Tests wether the predictions of the model are zero almost surely, which indicates that the model does not learn anything.
       Returns True if >98% of predictions are zero.
       Requires either 'data' or ('X_test' AND 'y_test').
    '''
    if data == None:
      predictions = pipeline.predict(X_test)
      if all(predictions == y_test):
        return False
    else:
      predictions = pipeline.predict(data.drop(['checked'], axis=1))
  
    zeros_fraction = np.mean(predictions == 0)
    return zeros_fraction > 0.98
    
  def testDrop(self, pipeline: Pipeline, data: pd.DataFrame, n_iter: int = 4, n_columns: int = 100):
    '''Implement a type of mutation testing.
        Trains the model with randomly dropped features. We expect performance to be worse than the default in all cases.
        n_iter: How many times to test.
        n_columns: How many features to drop every test.
    '''
    # Perform the test
    for i in range(n_iter):
        # Randomly select columns to drop (excluding the label column)
        columns_to_drop = np.random.choice(data.columns[:-1], size=n_columns, replace=False)

        # Drop the selected columns
        data_dropped = data.copy().drop(columns_to_drop, axis=1)

        # Split the data into features and labels
        X_mod = data_dropped.drop('checked', axis=1)
        y_mod = data_dropped['checked']
        X_mod = X_mod

        # Split the data into training and testing sets
        X_train_mod, X_test_mod, y_train_mod, y_test_mod = train_test_split(X_mod, y_mod, test_size=0.25, random_state=42)

        # Train the classifier
        pipeline.fit(X_train_mod, y_train_mod)

        # Make predictions on the test set
        y_pred_mod = pipeline.predict(X_test_mod)

        # Calculate accuracy
        accuracy = accuracy_score(y_test_mod, y_pred_mod)

        almostZero = self.testAlmostZero(pipeline, X_test=X_test_mod, y_test=y_test_mod)

        # Print the accuracy for this test case
        print(f"RUN {i + 1}: Accuracy with {n_columns} column(s) dropped: {accuracy:.4f}")
        print(f"RUN {i + 1}: Produces zero almost everywhere: {almostZero}")