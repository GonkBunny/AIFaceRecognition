from modAL.uncertainty import uncertainty_sampling, entropy_sampling;
import numpy as np
from math import log
from collections import Counter
from typing import Tuple

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax
from modAL.models.base import BaseCommittee


def count_votes(array):
    values, counts = np.unique(array, return_counts=True)
    
    votes = 0
    for i in counts:
        votes += i
    return votes


def entropy(array):
    # Assumir que log é de base e
    array_len = len(array)
    if array_len <= 1:
        return 0

    values, counts = np.unique(array, return_counts=True)
    probs = counts / array_len
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    entro = 0

    for i in probs:
        entro -= i * log(i)
    return entro


def votes_entropy(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs) -> np.ndarray:
    n_learners = len(committee)  # ver o número de membros no committee
    try:
        votes = committee.vote(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))
    X = np.asarray(X)
    p_vote = np.zeros(shape=(X.shape[0], len(committee.classes_)))
    entro = np.zeros(shape=(X.shape[0],))

    for vote_idx, vote in enumerate(votes):
        vote_counter = Counter(vote)  # criar dict com o numero de ocorrencias de cada classe

        for class_idx, class_label in enumerate(committee.classes_):
            p_vote[vote_idx, class_idx] = vote_counter[class_label] / n_learners
        entro[vote_idx] = entropy(p_vote[vote_idx])
    
    return entro


def vote_uncertain_sampling_entropy(committee: BaseCommittee, X: modALinput, n_instances: int = 1, r_tie_break=False, **disagreement_measure_kwargs):
    disagreement = votes_entropy(committee, X, **disagreement_measure_kwargs)

    if not r_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)
    X = np.asarray(X)
    return query_idx, X[query_idx]


def vote_uncertain_sampling_entropy_v2(committee: BaseCommittee, X: modALinput, n_instances: int = 5, r_tie_break=False, **disagreement_measure_kwargs):
    disagreement = votes_entropy(committee, X, **disagreement_measure_kwargs)

    if len(X) < n_instances:
        n_instances = len(X)
    if not r_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)

    pos_X_idx,pos_X = query_idx, X[query_idx]

    entropy,_ = entropy_sampling(committee,pos_X,1,False,**disagreement_measure_kwargs)

    return pos_X_idx[entropy], X[pos_X_idx[entropy]]


def votes(committee: BaseCommittee, X: modALinput, **predict_proba_kwargs):
     # ver o número de membros no committee
    try:
        votes = committee.vote(X, **predict_proba_kwargs)
    except NotFittedError:
        return np.zeros(shape=(X.shape[0],))
    X = np.asarray(X)
    
    learner = np.zeros(shape=(X.shape[0],))
    for vote_idx, vote in enumerate(votes):
        learner[vote_idx] = len(np.unique(vote))
    
    return learner


def vote_disagreement(committee: BaseCommittee, X: modALinput, n_instances: int = 1, r_tie_break=False, **disagreement_measure_kwargs):
    disagreement = votes(committee, X, **disagreement_measure_kwargs)

    if not r_tie_break:
        query_idx = multi_argmax(disagreement, n_instances=n_instances)
    else:
        query_idx = shuffled_argmax(disagreement, n_instances=n_instances)
    X = np.asarray(X)
    return query_idx, X[query_idx]

def random_choice(committee: BaseCommittee, X: modALinput, n_instances: int = 1, r_tie_break=False, **disagreement_measure_kwargs):
    query_idx = [np.random.randint(0, len(X))]
    return query_idx, X[query_idx]
