import os
import pickle
import numpy as np
import pandas as pd

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import OptimPreproc
from aif360.sklearn.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.algorithms.preprocessing import DisparateImpactRemover

from fairness.utils import set_working_dir
from fairness.utils import InvalidConfigException


class Mitigation:
    """A variety of algorithms can be used to mitigate bias.
    The choice of which to use depends on whether you want to fix the data (pre-process),
    the classifier (in-process), or the predictions (post-process)
    """
    def __init__(self, dataset, mitigation_config, dataset_working_dir):
        """
        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.
            unprivileged_groups (list(dict)): Representation for unprivileged group.
            privileged_groups (list(dict)): Representation for privileged group.
        """
        if not mitigation_config:
            raise InvalidConfigException("mitigation_config is not allocated. Execute Config.set_mitigation_config()")
        self.config = mitigation_config
        self.parent_dir = dataset_working_dir

        self.dataset = dataset
        self.unprivileged_groups = mitigation_config['unprivileged_groups']
        self.privileged_groups = mitigation_config['privileged_groups']

        self.working_dir = set_working_dir(self.parent_dir, str(self.config))

    def run(self, *args, **kwargs):
        algorithm = self.config['algorithm']
        new_dataset_path = os.path.join(self.working_dir, 'new_dataset.pickle')
        if os.path.exists(new_dataset_path):
            try:
                with open(new_dataset_path, 'rb') as fd:
                    new_dataset = pickle.load(fd)
                    res = self.convert_res(new_dataset, algorithm)
                return new_dataset, res
            except:
                pass

        new_dataset = self.__getattribute__(algorithm)(*args, **kwargs)
        res = self.convert_res(new_dataset, algorithm)
        # new_df, new_attr = new_dataset.convert_to_dataframe(de_dummy_code=True)

        with open(new_dataset_path, 'wb') as fd:
            pickle.dump(new_dataset, fd)

        return new_dataset, res

    def convert_res(self, dataset_transf, algorithm):
        if algorithm == 'reweighing':
            df_transf, attr_transf = dataset_transf.convert_to_dataframe(de_dummy_code=True)
            _df1 = df_transf[list(set([k for g in self.config['privileged_groups'] + self.config['unprivileged_groups'] for k in list(g.keys())])) + [self.config['label_name']]].reset_index(drop=True)
            _df2 = pd.DataFrame({'instance_weights': attr_transf['instance_weights']})
            instance_weights = pd.concat(
                [_df1, _df2], axis=1  # todo: convert privileged/unprivileged group to raw data(?)
            ).drop_duplicates()
            instance_weights = instance_weights.to_dict('records')
            return instance_weights
        else:
            return dataset_transf

    def reweighing(self):
        """Pre-processing
        Weights the examples in each (group, label) combination differently to ensure fairness before classification.
        ex) W_p_fav = P[exp](Privileged AND Favorable) / P[real](Privileged AND Favorable)
                = (P(Privileged) * P(Favorable)) / P(Privileged AND Favorable)
                = (n_p/n * n_fav/n) / (n_p_fav/n)
                = (n_p * n_fav) / (n * n_p_fav)
        """
        proc = Reweighing(
            self.unprivileged_groups,
            self.privileged_groups
        )
        dataset_transf = proc.fit(self.dataset).transform(self.dataset)
        return dataset_transf

    def optimized_preprocessing(self, optimizer, optim_options):
        """Pre-processing
        Learns a probabilistic transformation that can modify the features and the labels in the training data.

        Args:
            optimizer (class): Optimizer class.
            optim_options (dict): Options for optimization to estimate the transformation.
        """
        from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
        from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_german
        optim_options = {
            "distortion_fun": get_distortion_german,
            "epsilon": 0.05,
            "clist": [0.99, 1.99, 2.99],
            "dlist": [.1, 0.05, 0]
        }

        proc = OptimPreproc(
            optimizer=OptTools,
            optim_options=optim_options,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )
        dataset_transf = proc.fit(self.dataset, sep='=').transform(self.dataset, sep='=', transform_Y=True)
        return 'Not implemented yet.'

    def disparate_impact_remover(self):
        """Pre-processing"""
        DIs = []
        for level in np.linspace(0., 1., 11):
            di = DisparateImpactRemover(repair_level=level)
            repaired_dataset = di.fit_transform(self.dataset)

            cm = BinaryLabelDatasetMetric(
                repaired_dataset,
                privileged_groups=self.privileged_groups,
                unprivileged_groups=self.unprivileged_groups
            )
            disparate_impact = cm.disparate_impact()
            DIs.append(disparate_impact)
        return 'Not implemented yet.'

    def adversarial_debiasing(self, scope_name, sess):
        """In-processing
        Learns a classifier that maximizes prediction accuracy
        and simultaneously reduces an adversary's ability determine the protected attribute from te predictions.
        This approach leads to a fair classifier as the predictions cannot carry any group discrimination information
        that the adversary can exploit.

        Args:
            scope_name (str): scope name for the tenforflow variables
            sess (tf.Session): tensorflow session
        """
        proc = AdversarialDebiasing(
            self.unprivileged_groups,
            self.privileged_groups,
            scope_name,
            sess,
        )
        return "Not implemented yet"

    def calibrated_equalized_odds(self, cost_constraint='weighted'):
        """Post-processing"""
        proc = CalibratedEqOddsPostprocessing(
            self.unprivileged_groups,
            self.privileged_groups,
            cost_constraint=cost_constraint
        )
        """
        Args:
            dataset_true (BinaryLabelDataset): Dataset containing true `labels`. ~ ground truth
            dataset_pred (BinaryLabelDataset): Dataset containing predicted `scores`. ~ prediction score
        """
        dataset_new = proc.fit_predict(
            dataset_true=self.dataset,
            dataset_pred=None
        )
        return 'Not implemented yet.'

    def reject_option_based_classification(self, dataset_score):
        """Post-processing
        Changes predictions from a classifier to make them fairer.
        Provides favorable outcomes to unprivileged groups
        and unfavorable outcomes to privileged groups
        in a confidence band around the decision boundary with the highest uncertainty.

        Args:
            dataset_score (BinaryLabelDataset): Dataset containing the predicted `scores`.
        """
        proc = RejectOptionClassification(
            self.unprivileged_groups, self.privileged_groups,
        )
        dataset_new = proc.fit(self.dataset, dataset_score).predict(dataset_score)
        return dataset_new
