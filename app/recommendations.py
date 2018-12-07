"""Recommendation generation module."""

import logging
import numpy as np
import os
import pandas as pd
import pickle


logging.basicConfig(level=logging.INFO)

LOCAL_MODEL_PATH = '/Users/gurdasnijor/SideProjects/machine-learning/tensorflow-recommendation-wals/wals_ml_engine/jobs/wals_ml_local_20181206_170104'

ROW_MODEL_FILE = 'model/row.npy'
COL_MODEL_FILE = 'model/col.npy'
USER_MODEL_FILE = 'model/user.npy'
ITEM_MODEL_FILE = 'model/item.npy'
USER_ITEM_DATA_FILE = '/Users/gurdasnijor/SideProjects/machine-learning/tensorflow-recommendation-wals/data/recommendation_events.csv'

ITEM_PICKLE_PATH = '/Users/gurdasnijor/SideProjects/machine-learning/tensorflow-recommendation-wals/wals_ml_engine/mappings/itemmappings.pickle'
USER_PICKLE_PATH = '/Users/gurdasnijor/SideProjects/machine-learning/tensorflow-recommendation-wals/wals_ml_engine/mappings/usermappings.pickle'


class Recommendations(object):
  """Provide recommendations from a pre-trained collaborative filtering model.

  Args:
    local_model_path: (string) local path to model files
  """

  def __init__(self, local_model_path=LOCAL_MODEL_PATH):
    # _, project_id = google.auth.default()
    # self._bucket = 'recserve_' + project_id
    self._load_model(local_model_path)

  def _load_model(self, local_model_path):
    """Load recommendation model files from GCS.

    Args:
      local_model_path: (string) local path to model files
    """
    # load npy arrays for user/item factors and user/item maps
    self.user_factor = np.load(os.path.join(local_model_path, ROW_MODEL_FILE))
    self.item_factor = np.load(os.path.join(local_model_path, COL_MODEL_FILE))
    self.user_map = np.load(os.path.join(local_model_path, USER_MODEL_FILE))
    self.item_map = np.load(os.path.join(local_model_path, ITEM_MODEL_FILE))


    self.unique_items = pickle.load(open(ITEM_PICKLE_PATH, "rb"))
    self.unique_users = pickle.load(open(USER_PICKLE_PATH, "rb"))
    self.inv_map = {v: k for k, v in self.unique_items.items()}
    # self.inv_user_map = {v: k for k, v in self.unique_users.items()}


  def get_recommendations(self, user_id, num_recs):
    # generate list of recommended article indexes from model
    return self.destination_recommendations(user_id,
                                               self.user_factor,
                                               self.item_factor,
                                               num_recs)


  def destination_recommendations(self, workspace_slug, user_factor, item_factor, k):
    user_idx = self.unique_users[workspace_slug]
    # user_idx = np.searchsorted(self.user_map, user_idx)
    user_f = user_factor[user_idx]

    # dot product of item factors with user factor gives predicted ratings
    pred_ratings = item_factor.dot(user_f)

    # find candidate recommended item indexes sorted by predicted rating
    candidate_items = np.argsort(pred_ratings)[-k:]

    return list(map(lambda x: self.inv_map[x], candidate_items))


  def get_destination_map(self):
    print(self.unique_items)
    return self.unique_items

  def get_user_map(self):
    return self.unique_users


def generate_recommendations(user_idx, user_rated, row_factor, col_factor, k):
  """Generate recommendations for a user.

  Args:
    user_idx: the row index of the user in the ratings matrix,

    user_rated: the list of item indexes (column indexes in the ratings matrix)
      previously rated by that user (which will be excluded from the
      recommendations),

    row_factor: the row factors of the recommendation model

    col_factor: the column factors of the recommendation model

    k: number of recommendations requested

  Returns:
    list of k item indexes with the predicted highest rating,
    excluding those that the user has already rated
  """

  # bounds checking for args
  assert (row_factor.shape[0] - len(user_rated)) >= k

  print("ROW FACTOR", row_factor.size)
  print("COL FACTOR", col_factor)
  print("USER IDX", user_idx)

  # retrieve user factor
  user_f = row_factor[user_idx]

  # dot product of item factors with user factor gives predicted ratings
  pred_ratings = col_factor.dot(user_f)

  # find candidate recommended item indexes sorted by predicted rating
  k_r = k + len(user_rated)
  candidate_items = np.argsort(pred_ratings)[-k_r:]

  # remove previously rated items and take top k
  recommended_items = [i for i in candidate_items if i not in user_rated]
  recommended_items = recommended_items[-k:]

  # flip to sort highest rated first
  recommended_items.reverse()

  return recommended_items

