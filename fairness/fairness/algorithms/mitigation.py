import os
import pickle

from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing import OptimPreproc
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import RejectOptionClassification

from fairness.utils import set_working_dir


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
            raise "mitigation_config is not allocated. Execute Config.set_mitigation_config()"
        self.config = mitigation_config
        self.parent_dir = dataset_working_dir

        self.dataset = dataset
        self.unprivileged_groups = mitigation_config['unprivileged_groups']
        self.privileged_groups = mitigation_config['privileged_groups']

        self.working_dir = set_working_dir(self.parent_dir, str(self.config))

    def run(self, *args, **kwargs):
        algorithm = self.config['algorithm']
        # todo: exception
        # if algorithm not in []:
        #     raise ""
        new_dataset = self.__getattribute__(algorithm)(*args, **kwargs)
        # new_df, new_attr = new_dataset.convert_to_dataframe(de_dummy_code=True)

        new_dataset_path = os.path.join(self.working_dir, 'new_dataset.pickle')
        if os.path.exists(new_dataset_path):
            with open(new_dataset_path, 'rb') as fd:
                new_dataset = pickle.load(fd)
        else:
            with open(new_dataset_path, 'wb') as fd:
                pickle.dump(new_dataset, fd)

        return new_dataset

    def reweighing(self):
        """Pre-processing
        Weights the examples in each (group, label) combination differently to ensure fairness before classification.
        """
        proc = Reweighing(
            self.unprivileged_groups,
            self.privileged_groups
        )
        dataset_transf = proc.fit(self.dataset).transform(self.dataset)
        return dataset_transf

    def optimized_preprocessing(self, optimizer, optim_options):
        # todo: check args and test
        """Pre-processing
        Learns a probabilistic transformation that can modify the features and the labels in the training data.

        Args:
            optimizer (class): Optimizer class.
            optim_options (dict): Options for optimization to estimate the transformation.
        """
        proc = OptimPreproc(
            optimizer, optim_options,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups
        )
        dataset_transf = proc.fit(self.dataset, sep='=').transform(self.dataset, sep='=', transform_Y=True)
        return dataset_transf

    def adversarial_debiasing(self, scope_name, sess):
        # todo: check args and test
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
        dataset_new = proc.fit(self.dataset).predict(self.dataset)
        return dataset_new

    def reject_option_based_classification(self, dataset_score):
        # todo: check args and test
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
