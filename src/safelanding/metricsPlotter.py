import math
import random
from itertools import cycle, product
from pprint import pprint
from statistics import mean
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import (
    arange,
    argmax,
    argmin,
    concatenate,
    linspace,
    newaxis,
    unique,
    zeros_like,
)
from scipy import interp
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Code modifed from https://github.com/yohann84L/plot_metric


class Metrics:
    def __init__(self, test, prediction, threshold=0.5):
        self.test = test
        self.prediction = prediction
        self.threshold = threshold
        self.TP = 0
        self.FN = 0
        self.FP = 0
        self.TN = 0
        self._predict()

    def _predict(self):
        for i in range(len(self.test)):
            if self.test[i] == 0 and self.prediction[i] <= self.threshold:
                self.TN += 1
            elif self.test[i] == 1 and self.prediction[i] >= self.threshold:
                self.TP += 1
            elif self.test[i] == 1 and self.prediction[i] <= self.threshold:
                self.FN += 1
            elif self.test[i] == 0 and self.prediction[i] >= self.threshold:
                self.FP += 1

    def confusion_matrix(self):
        return np.array([[self.TP, self.FP], [self.FN, self.TN]])

    def specificity(self):
        try:
            return np.round(self.TN / (self.TN + self.FP), decimals=3)

        except:
            return None

    def sensitivity(self):
        try:
            return np.round(self.TP / (self.TP + self.FN), decimals=3)

        except:
            return None

    def false_positive_rate(self):
        try:
            return np.round(self.FP / (self.FP + self.TN), decimals=3)

        except:
            return None

    def precision(self):
        try:
            return np.round(self.TP / (self.TP + self.FP), decimals=3)

        except:
            return None

    def PRC_Value(self, number_threshold=100):
        precision = []
        recall = []
        for i in np.linspace(1.0, 0.0, num=number_threshold):
            aux = Metrics(self.test, self.prediction, threshold=i)
            precision_value = aux.precision()
            sensitivity_value = aux.sensitivity()
            if precision_value != None:  # Con un threshold de 1 no existen ni FP ni TP,
                # entonces precision: ZeroDivisionError-> salta el except y devuelve None

                precision.append(precision_value)
                recall.append(sensitivity_value)

        # precision, recall, _ = precision_recall_curve(self.test, self.prediction)
        return precision, recall

    def ROC_Value(self, number_threshold=100):
        recall = []
        false_positive_rate = []
        for i in np.linspace(1.0, 0.0, num=number_threshold):
            aux = Metrics(self.test, self.prediction, threshold=i)
            recall.append(aux.sensitivity())
            false_positive_rate.append(aux.false_positive_rate())

        return recall, false_positive_rate

    def auc(self, trapeze=False):  # Area-under-the-curve for ROC
        recall, false_positive_rate = self.ROC_Value()
        if trapeze is False:
            return (
                sum(
                    [
                        recall[i]
                        * (false_positive_rate[i + 1] - false_positive_rate[i])
                        for i in range(0, len(recall) - 1)
                    ]
                )
            ).round(3)
        else:
            return (np.trapz(recall, false_positive_rate)).round(3)
        # auc = roc_auc_score(self.test, self.prediction)

    def ap(self, trapeze=False):  # Area-under-the-curve for PRC
        precision, recall = self.PRC_Value()
        if trapeze is False:
            return (
                sum(
                    [
                        precision[i] * (recall[i + 1] - recall[i])
                        for i in range(0, len(precision) - 1)
                    ]
                )
            ).round(3)
        else:
            return (np.trapz(precision, recall)).round(3)

    def youden_index(self):
        return (self.sensitivity() + self.specificity() - 1).round(
            3
        )  # the optimum cut-off point, height above the chance line

    def f1(self):
        try:
            precision = self.precision()
            sensitivity = self.sensitivity()
            return (2 * ((precision * sensitivity) / (precision + sensitivity))).round(
                3
            )
        except:
            return None

class MetricsLz(Metrics):
    def __init__(self, df_lzs, threshold=0.5) -> None:
        self.y_pred = df_lzs["confidence"].tolist()
        self.y_gt = df_lzs["gt"].tolist()
        self.reasons = df_lzs["reasons"].tolist()
        self.reasonsFP = dict()
        self.lzs = df_lzs.to_dict("records")

        self.fpLz = []
        super().__init__(
            np.array(self.y_gt), np.array(self.y_pred), threshold=threshold
        )

    def _predict(self):
        for lz in self.lzs:
            if lz["gt"] == 0 and lz["confidence"] <= self.threshold:
                self.TN += 1
            elif lz["gt"] == 1 and lz["confidence"] >= self.threshold:
                self.TP += 1
            elif lz["gt"] == 1 and lz["confidence"] <= self.threshold:
                self.FN += 1
            elif lz["gt"] == 0 and lz["confidence"] >= self.threshold:
                self.FP += 1
                for reason in lz["reasons"]:
                    self.reasonsFP[reason] = self.reasonsFP.get(reason, 0) + 1
                    self.fpLz.append(lz)
class Optimum:
    def __init__(self, test, prediction):
        self.test = test
        self.prediction = prediction

    def optimum_by_youden(self):
        object_max = None
        threshold = None
        youden = 0
        for i in np.linspace(1.0, 0.0, num=100):
            aux = Metrics(self.test, self.prediction, threshold=i)
            youden_index = aux.youden_index()
            if youden_index > youden:
                youden = youden_index
                object_max = aux
                threshold = i
        return threshold.round(2), object_max

    def optimum_for_ROC(self):
        object_min = None
        threshold = None
        distance = math.sqrt(2)
        for i in np.linspace(1.0, 0.0, num=100):
            aux = Metrics(self.test, self.prediction, threshold=i)
            distance_aux = math.sqrt(
                (aux.false_positive_rate()) ** 2 + (aux.sensitivity() - 1) ** 2
            )
            if distance_aux < distance:
                distance = distance_aux
                object_min = aux
                threshold = i
        return threshold.round(2), object_min

    def optimum_for_PRC(self):
        object_min = None
        threshold = None
        distance = math.sqrt(2)
        for i in np.linspace(1.0, 0.0, num=100):
            aux = Metrics(self.test, self.prediction, threshold=i)
            if aux.precision() is not None:
                distance_aux = math.sqrt(
                    (aux.sensitivity() - 1) ** 2 + (aux.precision() - 1) ** 2
                )
                if distance_aux < distance:
                    distance = distance_aux
                    object_min = aux
                    threshold = i
        return threshold.round(2), object_min

    def optimum_by_f_score(self):
        object_max = None
        threshold = None
        f_score = 0  # F1 score reaches its best value at 1 and worst at 0
        for i in np.linspace(1.0, 0.0, num=100):
            aux = Metrics(self.test, self.prediction, threshold=i)
            f_score_aux = aux.f1()
            if f_score_aux != None and f_score_aux > f_score:
                f_score = f_score_aux
                object_max = aux
                threshold = i
        return threshold.round(2), object_max

    def optimum_by_sensitivity_specificity_difference(self):
        object_min = None
        threshold = None
        dif_sensitivity_specificity = []
        for i in np.linspace(1.0, 0.0, num=100):
            aux = Metrics(self.test, self.prediction, threshold=i)
            sensitivity_aux = aux.sensitivity()
            specificity_aux = aux.specificity()
            dif = abs(sensitivity_aux - specificity_aux)
            dif_sensitivity_specificity.append(dif)
            if dif == min(dif_sensitivity_specificity):
                object_min = aux
                threshold = i
        return threshold.round(2), object_min

    def optimum_by_recall_precision_difference(self):
        object_min = None
        threshold = None
        dif_recall_precision = []
        for i in np.linspace(1.0, 0.0, num=100):
            aux = Metrics(self.test, self.prediction, threshold=i)
            recall_aux = aux.sensitivity()
            precision_aux = aux.precision()
            if precision_aux != None:
                dif = abs(recall_aux - precision_aux)
                dif_recall_precision.append(dif)
                if dif == min(dif_recall_precision):
                    object_min = aux
                    threshold = i
        return threshold.round(2), object_min

    def report(self, colormap=False):
        threshold_youden, object_youden = self.optimum_by_youden()
        threshold_f_score, object_f_score = self.optimum_by_f_score()
        threshold_ROC, object_ROC = self.optimum_for_ROC()
        threshold_PRC, object_PRC = self.optimum_for_PRC()
        (
            threshold_difference_S_S,
            object_difference_S_S,
        ) = self.optimum_by_sensitivity_specificity_difference()
        (
            threshold_difference_R_P,
            object_difference_R_P,
        ) = self.optimum_by_recall_precision_difference()

        df = pd.DataFrame(
            {
                "Threshold": [
                    threshold_youden,
                    threshold_f_score,
                    threshold_ROC,
                    threshold_PRC,
                    threshold_difference_S_S,
                    threshold_difference_R_P,
                ],
                "TP": [
                    object_youden.TP,
                    object_f_score.TP,
                    object_ROC.TP,
                    object_PRC.TP,
                    object_difference_S_S.TP,
                    object_difference_R_P.TP,
                ],
                "TN": [
                    object_youden.TN,
                    object_f_score.TN,
                    object_ROC.TN,
                    object_PRC.TN,
                    object_difference_S_S.TN,
                    object_difference_R_P.TN,
                ],
                "FP": [
                    object_youden.FP,
                    object_f_score.FP,
                    object_ROC.FP,
                    object_PRC.FP,
                    object_difference_S_S.FP,
                    object_difference_R_P.FP,
                ],
                "FN": [
                    object_youden.FN,
                    object_f_score.FN,
                    object_ROC.FN,
                    object_PRC.FN,
                    object_difference_S_S.FN,
                    object_difference_R_P.FN,
                ],
                "Specificity": [
                    object_youden.specificity(),
                    object_f_score.specificity(),
                    object_ROC.specificity(),
                    object_PRC.specificity(),
                    object_difference_S_S.specificity(),
                    object_difference_R_P.specificity(),
                ],
                "Sensitivity": [
                    object_youden.sensitivity(),
                    object_f_score.sensitivity(),
                    object_ROC.sensitivity(),
                    object_PRC.sensitivity(),
                    object_difference_S_S.sensitivity(),
                    object_difference_R_P.sensitivity(),
                ],
                "PPV": [
                    object_youden.precision(),
                    object_f_score.precision(),
                    object_ROC.precision(),
                    object_PRC.precision(),
                    object_difference_S_S.precision(),
                    object_difference_R_P.precision(),
                ],
                "AUC": [object_youden.auc() for i in range(6)],
                "AP": [object_youden.ap() for i in range(6)],
            },
            index=[
                "Youden",
                "F-score",
                "Distance_ROC",
                "Distance_PRC",
                "Difference_Sensitivity_Specificity",
                "Difference_Recall_Precision",
            ],
        )
        if colormap:
            cm = sns.light_palette("green", as_cmap=True)
            s = df.sort_values(
                by="Threshold", ascending=False
            ).style.background_gradient(cmap=cm, high=0.5, low=-0.5, axis=0)
            # .set_properties(**{'width': '75px', 'text-align': 'center', 'font-size': '10pt'})
            return s

        return df



class BinaryClassification:
    """
    Initialize class.
    Parameters
    ----------
    y_true : array, list, shape = [n_sample]
        True binary labels.
    y_pred : array, list, shape = [n_sample]
        Target scores, can either be probability estimates of the positive
        class, confidence values, or non-thresholded measure of decisions
        (as returned by "decision_function" on some classifiers).
    labels : array, list, shape = [n_class]
        String or int of to define targeted classes.
    threshold : float [0-1], default=0.5,
        Classification threshold (or decision threshold).
        More information about threshold :
        - https://developers.google.com/machine-learning/crash-course/classification/thresholding
        - https://en.wikipedia.org/wiki/Threshold_model
    seaborn_style : string, default='darkgrid'
        Set the style of seaborn library, preset available with
        seaborn : darkgrid, whitegrid, dark, white, and ticks.
        See https://seaborn.pydata.org/tutorial/aesthetics.html#seaborn-figure-styles for more info.
    matplotlib_style : string, default=None
        Set the style of matplotlib. Find all preset here : https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
        Or with the following code :
    .. code:: python
        import matplotlib.style as style
        style.available
    """

    ### Parameters definition ###
    __param_precision_recall_curve = {
        "threshold": None,
        "plot_threshold": True,
        "beta": 1,
        "linewidth": 2,
        "fscore_iso": [0.2, 0.4, 0.6, 0.8],
        "iso_alpha": 0.7,
        "y_text_margin": 0.03,
        "x_text_margin": 0.2,
        "c_pr_curve": "black",
        "c_mean_prec": "red",
        "c_thresh": "black",
        "c_f1_iso": "grey",
        "c_thresh_point": "red",
        "ls_pr_curve": "-",
        "ls_mean_prec": "--",
        "ls_thresh": ":",
        "ls_fscore_iso": ":",
        "marker_pr_curve": None,
    }

    __param_confusion_matrix = {
        "threshold": None,
        "normalize": False,
        "title": "Confusion matrix",
        "cmap": plt.cm.Reds,
        "colorbar": True,
        "label_rotation": 45,
    }

    __param_roc_curve = {
        "threshold": None,
        "plot_threshold": True,
        "linewidth": 2,
        "y_text_margin": 0.05,
        "x_text_margin": 0.2,
        "c_roc_curve": "black",
        "c_random_guess": "red",
        "c_thresh_lines": "black",
        "ls_roc_curve": "-",
        "ls_thresh_lines": ":",
        "ls_random_guess": "--",
        "title": "Receiver Operating Characteristic",
        "loc_legend": "lower right",
    }

    __param_class_distribution = {
        "threshold": None,
        "display_prediction": True,
        "alpha": 0.5,
        "jitter": 0.3,
        "pal_colors": None,
        "display_violin": True,
        "c_violin": "white",
        "strip_marker_size": 4,
        "strip_lw_edge": None,
        "strip_c_edge": None,
        "ls_thresh_line": ":",
        "c_thresh_line": "red",
        "lw_thresh_line": 2,
        "title": None,
    }

    __param_threshold = {
        "threshold": None,
        "beta": 1,
        "title": None,
        "annotation": True,
        "bbox_dict": None,
        "bbox": True,
        "arrow_dict": None,
        "arrow": True,
        "plot_fscore": True,
        "plot_recall": True,
        "plot_prec": True,
        "plot_fscore_max": True,
        "c_recall_line": "green",
        "lw_recall_line": 2,
        "ls_recall_line": "-",
        "label_recall": "Recall",
        "marker_recall": "",
        "c_prec_line ": "blue",
        "lw_prec_line": 2,
        "ls_prec_line": "-",
        "label_prec": "Precision",
        "marker_prec": "",
        "c_fscr_line ": "red",
        "lw_fscr_line": 2,
        "ls_fscr_line": "-",
        "label_fscr": None,
        "marker_fscr": "",
        "marker_fscore_max": "o",
        "c_fscore_max": "red",
        "markersize_fscore_max": 5,
        "plot_threshold": True,
        "c_thresh_line": "black",
        "lw_thresh_line": 2,
        "ls_thresh_line": "--",
        "plot_best_threshold": True,
        "c_bestthresh_line": "black",
        "lw_bestthresh_line": 1,
        "ls_bestthresh_line": ":",
    }

    def __init__(
        self,
        y_true,
        y_pred,
        labels,
        threshold=0.5,
        seaborn_style="darkgrid",
        matplotlib_style=None,
    ):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.threshold = threshold
        sns.set_style(seaborn_style)
        if matplotlib_style is not None:
            style.use("ggplot")

    def get_function_parameters(self, function, as_df=False):
        """
        Function to get all available parameters for a given function.
        Parameters
        ----------
        function : func
            Function parameter's wanted.
        as_df : boolean, default=False
            Set to True to return a dataframe with parameters instead of dictionnary.
        Returns
        -------
        param_dict : dict
            Dictionnary containing parameters for the given function and their default value.
        """

        if function.__name__ == "plot_precision_recall_curve":
            param_dict = self.__param_precision_recall_curve
        elif function.__name__ == "plot_confusion_matrix":
            param_dict = self.__param_confusion_matrix
        elif function.__name__ == "plot_roc_curve":
            param_dict = self.__param_roc_curve
        elif function.__name__ == "plot_class_distribution":
            param_dict = self.__param_class_distribution
        elif function.__name__ == "plot_threshold":
            param_dict = self.__param_threshold
        else:
            print("Wrong function given, following functions are available : ")
            for func in filter(
                lambda x: callable(x), BinaryClassification.__dict__.values()
            ):
                print(func.__name__)
            return [
                func()
                for func in filter(
                    lambda x: callable(x), BinaryClassification.__dict__.values()
                )
            ]

        if as_df:
            return pd.DataFrame.from_dict(param_dict, orient="index")
        else:
            return param_dict

    def plot_confusion_matrix(
        self,
        threshold=None,
        normalize=False,
        title="Confusion matrix",
        cmap=plt.cm.Reds,
        colorbar=True,
        label_rotation=45,
    ):
        """
        Plots the confusion matrix.
        Parameters
        ----------
        threshold : float, default=0.5
            Threshold to determnine the rate between positive and negative values of the classification.
        normalize : bool, default=False
            Set to True to normalize matrix and make matrix coefficient between 0 and 1.
        title : string, default="Confusion matrix",
            Set title of the plot.
        cmap : colormap, default=plt.cm.Reds
            Colormap of the matrix. See https://matplotlib.org/examples/color/colormaps_reference.html to find all
            available colormap.
        colorbar : bool, default=True
            Display color bar beside matrix.
        label_rotation : int, default=45
            Degree of rotation for x_axis labels.
        Returns
        -------
        cm : array, shape=[n_classes, n_classes]
            Return confusion_matrix computed by sklearn.metrics.confusion_matrix
        """
        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        # Convert prediction probility into class
        y_pred_class = [1 if y_i > t else 0 for y_i in self.y_pred]

        # Define the confusion matrix
        cm = confusion_matrix(self.y_true, y_pred_class, labels=[0, 1])

        # Normalize matrix if choosen
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, newaxis]
            title = title + " normalized"

        # Compute plot
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        if colorbar:
            plt.colorbar()
        tick_marks = arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=label_rotation)
        plt.yticks(tick_marks, self.labels)

        # Display text into matrix
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        return cm

    def plot_roc_curve(
        self,
        threshold=None,
        plot_threshold=True,
        linewidth=2,
        y_text_margin=0.05,
        x_text_margin=0.2,
        c_roc_curve="black",
        c_random_guess="red",
        c_thresh_lines="black",
        ls_roc_curve="-",
        ls_thresh_lines=":",
        ls_random_guess="--",
        title="Receiver Operating Characteristic",
        loc_legend="lower right",
    ):
        """
        Compute and plot the ROC (Receiver Operating Characteristics) curve but also AUC (Area Under The Curve).
        Note : for more information about ROC curve and AUC look at the reference given.
        Moreover, this implementation is restricted to binary classification only.
        See MultiClassClassification for multi-classes implementation.
        Parameters
        ----------
        threshold : float, default=0.5
        plot_threshold : boolean, default=True
            Plot or not ROC lines for the given threshold.
        linewidth : float, default=2
        y_text_margin : float, default=0.03
            Margin (y) of text threshold.
        x_text_margin : float, default=0.2
            Margin (x) of text threshold.
        c_roc_curve : string, default='black'
            Define the color of ROC curve.
        c_random_guess : string, default='red'
            Define the color of random guess line.
        c_thresh_lines : string, default='black'
            Define the color of threshold lines.
        ls_roc_curve : string, default='-'
            Define the linestyle of ROC curve.
        ls_thresh_lines : string, default=':'
            Define the linestyle of threshold lines.
        ls_random_guess : string, default='--'
            Define the linestyle of random guess line.
        title : string, default='Receiver Operating Characteristic'
            Set title of the figure.
        loc_legend : string, default='loc_legend'
            Localisation of legend. Available string are the following :
            ================    ================
            Location String	    Location Code
            ================    ================
            'best'	            0
            'upper right'	    1
            'upper left'	    2
            'lower left'	    3
            'lower right'	    4
            'right'         	5
            'center left'	    6
            'center right'	    7
            'lower center'	    8
            'upper center'	    9
            'center'	        10
            ================    ================
        Returns
        -------
        fpr : array, shape = [>2]
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= thresholds[i].
        tpr : array, shape = [>2]
            Increasing true positive rates such that element i is the true
            positive rate of predictions with score >= thresholds[i].
        thresh : array, shape = [n_thresholds]
            Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
        auc : float
        References
        -------
        .. [1] `Understanding AUC - ROC Curve (article by Sarang Narkhede)
            <https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5>`_
        .. [2] `Wikipedia entry for the Receiver operating characteristic
            <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
        .. [3] `sklearn documentation about roc_curve and auc functions
            <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html>`_
        """
        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        # Compute ROC Curve
        fpr, tpr, thresh = roc_curve(self.y_true, self.y_pred)
        # Compute AUC
        roc_auc = auc(fpr, tpr)

        # Compute the y & x axis to trace the threshold
        idx_thresh, idy_thresh = (
            fpr[argmin(abs(thresh - t))],
            tpr[argmin(abs(thresh - t))],
        )

        # Plot roc curve
        plt.plot(
            fpr,
            tpr,
            color=c_roc_curve,
            lw=linewidth,
            label="ROC curve (area = %0.2f)" % roc_auc,
            linestyle=ls_roc_curve,
        )

        # Plot reference line
        plt.plot(
            [0, 1],
            [0, 1],
            color=c_random_guess,
            lw=linewidth,
            linestyle=ls_random_guess,
            label="Random guess",
        )
        # Plot threshold
        if plot_threshold:
            # Plot vertical and horizontal line
            plt.axhline(
                y=idy_thresh,
                color=c_thresh_lines,
                linestyle=ls_thresh_lines,
                lw=linewidth,
            )
            plt.axvline(
                x=idx_thresh,
                color=c_thresh_lines,
                linestyle=ls_thresh_lines,
                lw=linewidth,
            )

            # Plot text threshold
            if idx_thresh > 0.5 and idy_thresh > 0.5:
                plt.text(
                    x=idx_thresh - x_text_margin,
                    y=idy_thresh - y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )
            elif idx_thresh <= 0.5 and idy_thresh <= 0.5:
                plt.text(
                    x=idx_thresh + x_text_margin,
                    y=idy_thresh + y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )
            elif idx_thresh <= 0.5 < idy_thresh:
                plt.text(
                    x=idx_thresh + x_text_margin,
                    y=idy_thresh - y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )
            elif idx_thresh > 0.5 >= idy_thresh:
                plt.text(
                    x=idx_thresh - x_text_margin,
                    y=idy_thresh + y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )

            # Plot redpoint of threshold on the ROC curve
            plt.plot(idx_thresh, idy_thresh, "ro")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc=loc_legend)

        return fpr, tpr, thresh, roc_auc

    def plot_precision_recall_curve(
        self,
        threshold=None,
        plot_threshold=True,
        beta=1,
        linewidth=2,
        fscore_iso=[0.2, 0.4, 0.6, 0.8],
        iso_alpha=0.7,
        y_text_margin=0.03,
        x_text_margin=0.2,
        c_pr_curve="black",
        c_mean_prec="red",
        c_thresh_lines="black",
        c_f1_iso="grey",
        c_thresh_point="red",
        ls_pr_curve="-",
        ls_mean_prec="--",
        ls_thresh=":",
        ls_fscore_iso=":",
        marker_pr_curve=None,
        title="Precision and Recall Curve",
    ):
        """
        Compute and plot the precision-recall curve.
        Note : this implementation is restricted to binary classification only.
        See MultiClassClassification for multi-classes implementation.
        F1-iso are curve where a given f1-score is constant.
        We also consider the use of F_beta-score, change the parameter beta to use an other f-score.
        "Two other commonly used F measures are the F_2 measure, which weighs recall higher than
        precision (by placing more emphasis on false negatives), and the F_0.5 measure, which weighs
        recall lower than precision (by attenuating the influence of false negatives). (Wiki)"
        Parameters
        ----------
        threshold : float, default=0.5
            Threshold to determnine the rate between positive and negative values of the classification.

        plot_threshold : boolean, default=True
            Plot or not precision and recall lines for the given threshold.

        beta : float, default=1,
            Set beta to another float to use a different f_beta score. See definition of f_beta-score
            for more information : https://en.wikipedia.org/wiki/F1_score

        linewidth : float, default=2

        fscore_iso : array, list, default=[0.2, 0.4, 0.6, 0.8]
            List of float f1-score. Set to None or empty list to remove plotting of iso.

        iso_alpha : float, default=0.7
            Transparency of iso-f1.

        y_text_margin : float, default=0.03
            Margin (y) of text threshold.
        x_text_margin : float, default=0.2
            Margin (x) of text threshold.

        c_pr_curve : string, default='black'
            Define the color of precision-recall curve.

        c_mean_prec : string, default='red'
            Define the color of mean precision line.

        c_thresh : string, default='black'
            Define the color of threshold lines.

        c_f1_iso : string, default='grey'
            Define the color of iso-f1 curve.

        c_thresh_point : string, default='red'
            Define the color of threshold point.

        ls_pr_curve : string, default='-'
            Define the linestyle of precision-recall curve.

        ls_mean_prec : string, default='--'
            Define the linestyle of mean precision line.

        ls_thresh : string, default=':'
            Define the linestyle of threshold lines.

        ls_fscore_iso : string, default=':'
            Define the linestyle of iso-f1 curve.

        marker_pr_curve : string, default=None
            Define the marker of precision-recall curve.

        title : string, default="Precision and Recall Curve"
            Set title of the figure.
        Returns
        -------
        prec : array, shape = [n_thresholds + 1]
            Precision values such that element i is the precision of
            predictions with score >= thresholds[i] and the last element is 1.

        recall : array, shape = [n_thresholds + 1]
            Decreasing recall values such that element i is the recall of
            predictions with score >= thresholds[i] and the last element is 0.

        thresh : array, shape = [n_thresholds <= len(np.unique(y_pred))]
            Increasing thresholds on the decision function used to compute
            precision and recall.
        """

        # Set f1-iso and threshold parameters
        if fscore_iso == None:
            fscore_iso = []
        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        # List for legends
        lines, labels = [], []

        # Compute precision and recall
        prec, recall, thresh = precision_recall_curve(self.y_true, self.y_pred)
        # Compute area
        pr_auc = average_precision_score(self.y_true, self.y_pred)
        # Compute the y & x axis to trace the threshold
        idx_thresh, idy_thresh = (
            recall[argmin(abs(thresh - t))],
            prec[argmin(abs(thresh - t))],
        )

        # Plot PR curve
        (l,) = plt.plot(
            recall,
            prec,
            color=c_pr_curve,
            lw=linewidth,
            linestyle=ls_pr_curve,
            marker=marker_pr_curve,
        )
        lines.append(l)
        labels.append("PR curve (area = {})".format(round(pr_auc, 2)))

        # Plot mean precision
        (l,) = plt.plot(
            [0, 1],
            [mean(prec), mean(prec)],
            color=c_mean_prec,
            lw=linewidth,
            linestyle=ls_mean_prec,
        )
        lines.append(l)
        labels.append("Mean precision = {}".format(round(mean(prec), 2)))

        # Fscore-iso
        if len(fscore_iso) > 0:  # Check to plot or not the fscore-iso
            for f_score in fscore_iso:
                x = linspace(0.005, 1, 100)  # Set x range
                y = (
                    f_score * x / (beta ** 2 * x + x - beta ** 2 * f_score)
                )  # Compute fscore-iso using f-score formula
                (l,) = plt.plot(
                    x[y >= 0],
                    y[y >= 0],
                    color=c_f1_iso,
                    linestyle=ls_fscore_iso,
                    alpha=iso_alpha,
                )
                plt.text(
                    s="f{:s}={:0.1f}".format(str(beta), f_score),
                    x=0.9,
                    y=y[-10] + 0.02,
                    alpha=iso_alpha,
                )
            lines.append(l)
            labels.append("iso-f{:s} curves".format(str(beta)))

            # Set ylim to see entire iso and to avoid a max ylim really high
            plt.ylim([0.0, 1.05])

        # Plot threshold
        if plot_threshold:
            # Plot vertical and horizontal line
            plt.axhline(
                y=idy_thresh, color=c_thresh_lines, linestyle=ls_thresh, lw=linewidth
            )
            plt.axvline(
                x=idx_thresh, color=c_thresh_lines, linestyle=ls_thresh, lw=linewidth
            )

            # Plot text threshold
            if idx_thresh > 0.5 and idy_thresh > 0.5:
                plt.text(
                    x=idx_thresh - x_text_margin,
                    y=idy_thresh - y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )
            elif idx_thresh <= 0.5 and idy_thresh <= 0.5:
                plt.text(
                    x=idx_thresh + x_text_margin,
                    y=idy_thresh + y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )
            elif idx_thresh <= 0.5 < idy_thresh:
                plt.text(
                    x=idx_thresh + x_text_margin,
                    y=idy_thresh - y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )
            elif idx_thresh > 0.5 >= idy_thresh:
                plt.text(
                    x=idx_thresh - x_text_margin,
                    y=idy_thresh + y_text_margin,
                    s="Threshold : {:.2f}".format(t),
                )

            # Plot redpoint of threshold on the ROC curve
            plt.plot(idx_thresh, idy_thresh, marker="o", color=c_thresh_point)

        # Axis and legends
        plt.xlim([0.0, 1.0])
        plt.legend(lines, labels)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        if plot_threshold:
            plt.title("{} (Threshold = {})".format(title, round(t, 2)))
        else:
            plt.title(title)

        return prec, recall, thresh

    def plot_class_distribution(
        self,
        threshold=None,
        display_prediction=True,
        alpha=0.5,
        jitter=0.3,
        pal_colors=None,
        display_violin=True,
        c_violin="white",
        strip_marker_size=4,
        strip_lw_edge=None,
        strip_c_edge=None,
        ls_thresh_line=":",
        c_thresh_line="red",
        lw_thresh_line=2,
        title=None,
    ):
        """
        Plot distribution of the predictions for each classes.
        Note : Threshold here is importante because it define colors for True Positive,
        False Negative, True Nagative and False Positive.
        Parameters
        ----------
        threshold : float, default=0.5
            Threshold to determnine the rate between positive and negative values of the classification.
        display_prediction : bool, default=True
            Display the point representing each predictions.
        alpha : float, default=0.5
            Transparency of each predicted point.
        jitter : float, default=0.3
                Amount of jitter (only along the categorical axis) to apply. This can be useful when you have many
                points and they overlap, so that it is easier to see the distribution. You can specify the amount
                of jitter (half the width of the uniform random variable support), or just use True for a good default.
                See : https://seaborn.pydata.org/generated/seaborn.stripplot.html
        pal_colors : palette name, list, or dict, optional, default=["#00C853", "#FF8A80", "#C5E1A5", "#D50000"]
            Colors to use for the different levels of the hue variable. Should be something that can be interpreted
            by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
            See : https://seaborn.pydata.org/generated/seaborn.stripplot.html
        display_violin : bool, default=True
            Display violin plot.
        c_violin : string, default='white'
            Color of the violinplot.
        strip_marker_size : int, default='4'
            Size of marker representing predictions.
        strip_lw_edge : float, default=None
            Size of the linewidth for the edge of point prediction.
        strip_c_edge : string, default=None
            Color of the linewidth for the edge of point prediction.
        ls_thresh_line : string, default=':'
            Linestyle for the threshold line.
        c_thresh_line : string, default='red'
            Color for the threshold line.
        lw_thresh_line : float, default=2
            Line width of the threshold line.
        title : string, default=None
            String for the title of the graphic.
        Returns
        -------
        DataFrame with the following column :
        - True Class
        - Predicted Proba
        - Predicted Type
        - Predicted Class
        """
        if pal_colors == None:
            pal_colors = ["#00C853", "#FF8A80", "#C5E1A5", "#D50000"]
        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        def __compute_thresh(row, _threshold):
            if (row["pred"] >= _threshold) & (row["class"] == 1):
                return "TP"
            elif (row["pred"] >= _threshold) & (row["class"] == 0):
                return "FP"
            elif (row["pred"] < _threshold) & (row["class"] == 1):
                return "FN"
            elif (row["pred"] < _threshold) & (row["class"] == 0):
                return "TN"

        pred_df = pd.DataFrame({"class": self.y_true, "pred": self.y_pred})

        pred_df["type"] = pred_df["pred"]
        pred_df["type"] = pred_df.apply(lambda x: __compute_thresh(x, t), axis=1)

        # Plot violin pred distribution
        if display_violin:
            sns.violinplot(
                x="class", y="pred", data=pred_df, inner=None, color=c_violin, cut=0
            )

        # Plot prediction distribution
        if display_prediction:
            sns.stripplot(
                x="class",
                y="pred",
                hue="type",
                data=pred_df,
                jitter=jitter,
                alpha=alpha,
                size=strip_marker_size,
                palette=sns.color_palette(pal_colors),
                linewidth=strip_lw_edge,
                edgecolor=strip_c_edge,
            )

        # Plot threshold
        plt.axhline(
            y=t, color=c_thresh_line, linewidth=lw_thresh_line, linestyle=ls_thresh_line
        )
        if title == None:
            plt.title("Threshold at {:.2f}".format(t))
        else:
            plt.title(title)

        pred_df["Predicted Class"] = pred_df["pred"].apply(
            lambda x: self.labels[1] if x >= t else self.labels[0]
        )
        pred_df.columns = [
            "True Class",
            "Predicted Proba",
            "Predicted Type",
            "Predicted Class",
        ]
        return pred_df

    def plot_score_distribution(
        self,
        threshold=None,
        plot_hist_TN=True,
        kde_ksw_TN={"shade": True},
        label_TN="True Negative",
        c_TN_curve="green",
    ):
        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        df = pd.DataFrame({"y_true": self.y_true, "y_pred": self.y_pred})

        TN_pred = list(df[df["y_true"] == 0]["y_pred"])
        TP_pred = list(df[df["y_true"] == 1]["y_pred"])

        # Plot True Negative predictions
        ax = sns.distplot(
            TN_pred,
            hist=plot_hist_TN,
            kde_kws=kde_ksw_TN,
            label=label_TN,
            color=c_TN_curve,
        )
        # Plot False Positive predictions using hatch
        kde_x, kde_y = ax.lines[0].get_data()
        ax.fill_between(
            kde_x,
            kde_y,
            where=(kde_x >= t),
            interpolate=True,
            facecolor="none",
            hatch="////",
            edgecolor="black",
            label="False Positive",
        )

        # Plot True Positive predictions
        ax = sns.distplot(
            TP_pred,
            hist=False,
            color="r",
            kde_kws={"shade": True},
            label="True Positive",
        )
        # Plot False Negative predictions using hatch
        kde_x, kde_y = ax.lines[1].get_data()
        ax.fill_between(
            kde_x,
            kde_y,
            where=(kde_x <= t),
            interpolate=True,
            facecolor="none",
            hatch="\\\\\\\\",
            edgecolor="black",
            label="False Negative",
        )

        # Plot the threshold
        plt.axvline(t, label="Threshold {:.2f}".format(t), color="black", linestyle=":")

        # Show legend
        plt.legend(loc="best")
        # Set axis and title
        plt.xlabel("Predictions probability")
        plt.xlabel("Predicted observations")
        plt.title("Distribution of predicted probability")
        plt.xlim(0, 1)

    def plot_threshold(
        self,
        threshold=None,
        beta=1,
        title=None,
        annotation=True,
        bbox_dict=None,
        bbox=True,
        arrow_dict=None,
        arrow=True,
        plot_fscore=True,
        plot_recall=True,
        plot_prec=True,
        plot_fscore_max=True,
        c_recall_line="green",
        lw_recall_line=2,
        ls_recall_line="-",
        label_recall="Recall",
        marker_recall="",
        c_prec_line="blue",
        lw_prec_line=2,
        ls_prec_line="-",
        label_prec="Precision",
        marker_prec="",
        c_fscr_line="red",
        lw_fscr_line=2,
        ls_fscr_line="-",
        label_fscr=None,
        marker_fscr="",
        marker_fscore_max="o",
        c_fscore_max="red",
        markersize_fscore_max=5,
        plot_threshold=True,
        c_thresh_line="black",
        lw_thresh_line=2,
        ls_thresh_line="--",
        plot_best_threshold=True,
        c_bestthresh_line="black",
        lw_bestthresh_line=1,
        ls_bestthresh_line=":",
    ):
        """
        Plot precision - threshold, recall - threshold and fbeta-score - threshold curves.
        Also plot threshold line for a given threshold and threshold line for the best ratio between precision
        and recall.
        Parameters
        ----------
        threshold : float, default=0.5
            Threshold to determnine the rate between positive and negative values of the classification.
        beta : float, default=1,
            Set beta to another float to use a different f_beta score. See definition of f_beta-score
            for more information : https://en.wikipedia.org/wiki/F1_score
        title : string, default=None
            String for the title of the graphic.
        annotation : bool, default=True
            Boolean to display annotation box with theshold, precision and recall score.
        bbox_dict : dict, default={'facecolor': 'none',
                                'edgecolor': 'black',
                                'boxstyle': 'round',
                                'alpha': 0.4,
                                'pad': 0.3}
            Set the parameters of the bbox annotation. See matplotlib documentation_ for more information.
        bbox : bool, default=True
            Boolean to display the bbox around annotation.
        arrow_dict : dict, default={'arrowstyle': "->", 'color': 'black'}
            Set the parameters of the bbox annotation. See matplotlib documentation_ for more information.
        arrow : bool, default=True
            Boolean to display the array for the annotation.
        plot_fscore : bool, default=True
            Boolean to plot the FBeta-Score curve.
        plot_recall : bool, default=True
            Boolean to plot the recall curve.
        plot_prec : bool, default=True
            Boolean to plot the precision curve.
        plot_fscore_max : bool, default=True
            Boolean to plot the point showing fbeta-score max.
        c_recall_line : string, default='green'
            Color of the recall curve.
        lw_recall_line : float, default=2
            Linewidth of the recall curve.
        ls_recall_line : string, default='-'
            Linestyle of the recall curve.
        label_recall : string, default='Recall'
            Label of the recall curve.
        marker_recall : string, default=''
            Marker of the recall curve.
        c_prec_line : string, default='green'
            Color of the prec curve.
        lw_prec_line : float, default=2
            Linewidth of the prec curve.
        ls_prec_line : string, default='-'
            Linestyle of the prec curve.
        label_prec : string, default='prec'
            Label of the prec curve.
        marker_prec : string, default=''
            Marker of the prec curve.
        c_fscr_line : string, default='green'
            Color of the fscr curve.
        lw_fscr_line : float, default=2
            Linewidth of the fscr curve.
        ls_fscr_line : string, default='-'
            Linestyle of the fscr curve.
        label_fscr : string, default='fscr'
            Label of the fscr curve.
        marker_fscr : string, default=''
            Marker of the fscr curve.
        marker_fscore_max : string, default='o'
            Marker for the fscore max point.
        c_fscore_max : string, default='red'
            Color for the fscore max point.
        markersize_fscore_max : float, default=5
            Marker size for the fscore max point.
        plot_threshold : bool, default=True
            Plot a line at the given threshold.
        c_thresh_line : string, default='black'
            Color for the threshold line.
        lw_thresh_line : float, default=2
            Linewidth for the threshold line.
        ls_thresh_line : string, default='--'
            Linestyle for the threshold line.
        plot_best_threshold : bool, default=True
            Plot a line at the best threshold (best ratio precision-recall).
        c_bestthresh_line : string, default='black'
            Color for the best threshold line.
        lw_bestthresh_line : float, default=2
            Linewidth for the best threshold line.
        ls_bestthresh_line : string, default='--'
            Linestyle for the best threshold line.
        Returns
        -------
        References
        ----------
        .. _documentation: https://matplotlib.org/users/annotations.html#annotating-with-text-with-box
        """
        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred)
        fscore = (
            (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall)
        )

        thresh = linspace(0, 1, len(recall))
        y_max_fscore, x_max_fscore = max(fscore), thresh[argmax(fscore)]

        opti_thresh = 0
        opti_recall = 0
        for i, t_ in enumerate(thresh):
            if abs(precision[i] - recall[i]) < 0.01:
                opti_thresh = t_
                opti_preci = precision[i]
                opti_recall = recall[i]
                break

        # Plot recall
        if plot_recall:
            plt.plot(
                thresh,
                recall,
                label=label_recall,
                color=c_recall_line,
                lw=lw_recall_line,
                linestyle=ls_recall_line,
                marker=marker_recall,
            )

        # Plot precision
        if plot_prec:
            plt.plot(
                thresh,
                precision,
                label=label_prec,
                color=c_prec_line,
                lw=lw_prec_line,
                linestyle=ls_prec_line,
                marker=marker_prec,
            )

        # Plot fbeta-score
        if plot_fscore:
            if label_fscr == None:
                label_fscr = "F{:s}-score (max={:.03f})".format(str(beta), y_max_fscore)
            plt.plot(
                thresh,
                fscore,
                label=label_fscr,
                color=c_fscr_line,
                lw=lw_fscr_line,
                linestyle=ls_fscr_line,
                marker=marker_fscr,
            )

        # Plot max fbeta-score
        if plot_fscore_max:
            plt.plot(
                x_max_fscore,
                y_max_fscore,
                marker=marker_fscore_max,
                markersize=markersize_fscore_max,
                color=c_fscore_max,
            )

        # Plot threshold
        if plot_threshold:
            plt.axvline(
                t, linestyle=ls_thresh_line, color=c_thresh_line, lw=lw_thresh_line
            )
        if plot_best_threshold:
            plt.axvline(
                opti_thresh,
                linestyle=ls_bestthresh_line,
                color=c_bestthresh_line,
                lw=lw_bestthresh_line,
            )
            plt.plot(
                opti_thresh,
                opti_recall,
                color=c_bestthresh_line,
                marker="o",
                markersize=4,
            )
        # Plot best rate between prec/recall
        if annotation:
            ## Annotation dict :
            if bbox == True and bbox_dict == None:
                bbox_dict = dict(
                    facecolor="none",
                    edgecolor="black",
                    boxstyle="round",
                    alpha=0.4,
                    pad=0.3,
                )
            if arrow == True and arrow_dict == None:
                arrow_dict = dict(arrowstyle="->", color="black")
            plt.annotate(
                s="Thresh = {:0.2f}\nRecall=Prec={:0.2f}".format(
                    opti_thresh, opti_recall
                ),
                xy=(opti_thresh, opti_recall),
                xytext=(opti_thresh + 0.02, opti_recall - 0.2),
                bbox=bbox_dict,
                arrowprops=arrow_dict,
            )

        # Limit
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Plot text
        if title == None:
            plt.title(
                "Precision/Recall/F{:s}-score - Threshold Curve".format(str(beta))
            )
        else:
            plt.title(title)
        plt.xlabel("Threshold")
        plt.legend()
        plt.xticks(arange(0, 1, 0.1))
        plt.ylabel("Scores")

    def print_report(self, threshold=0.5):
        from sklearn.metrics import classification_report

        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        y_pred_class = [1 if y_i > t else 0 for y_i in self.y_pred]

        print("                   ________________________")
        print("                  |  Classification Report |")
        print("                   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
        print(
            classification_report(
                self.y_true, y_pred_class, target_names=list(map(str, self.labels))
            )
        )


class MultiClassClassification:
    def __init__(self, y_true, y_pred, labels, threshold=0.5):
        """Constructor of the class"""
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.threshold = threshold
        self.n_classes = len(labels)

    def __to_hex(self, scale):
        """converts scale of rgb or hsl strings to list of tuples with rgb integer values. ie,
        [ "rgb(255, 255, 255)", "rgb(255, 255, 255)", "rgb(255, 255, 255)" ] -->
        [ (255, 255, 255), (255, 255, 255), (255, 255, 255) ]"""
        numeric_scale = []
        for s in scale:
            s = s[s.find("(") + 1 : s.find(")")].replace(" ", "").split(",")
            numeric_scale.append((float(s[0]), float(s[1]), float(s[2])))

        return ["#%02x%02x%02x" % tuple(map(int, s)) for s in numeric_scale]

    def plot_roc(self, threshold=None, linewidth=2, show_threshold=False):
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        if threshold == None:
            t = self.threshold
        else:
            t = threshold

        # Binarize the output
        y = label_binarize(self.y_true, classes=self.labels)

        # Compute ROC curve and ROC area for each class
        fpr, tpr = dict(), dict()
        roc_auc = dict()
        idx_thresh, idy_thresh = dict(), dict()
        for i in range(self.n_classes):
            fpr[i], tpr[i], thresh = roc_curve(y[:, i], self.y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute the y & x axis to trace the threshold
            idx_thresh[i], idy_thresh[i] = (
                fpr[i][argmin(abs(thresh - t))],
                tpr[i][argmin(abs(thresh - t))],
            )

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], thresh = roc_curve(y.ravel(), self.y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        idx_thresh["micro"] = fpr["micro"][argmin(abs(thresh - t))]
        idy_thresh["micro"] = tpr["micro"][argmin(abs(thresh - t))]

        # Aggregate all false positive rates
        all_fpr = unique(concatenate([fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Average it and compute AUC
        mean_tpr /= self.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=linewidth,
        )

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})"
            "".format(roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=linewidth,
        )

        random.seed(124)
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"]
        random.shuffle(colors)
        for i, color in zip(range(self.n_classes), cycle(colors)):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=linewidth,
                alpha=0.5,
                label="ROC curve of class {0} (area = {1:0.2f})"
                "".format(i, roc_auc[i]),
            )

        if show_threshold:
            plt.plot(idx_thresh.values(), idy_thresh.values(), "ro")

        plt.plot([0, 1], [0, 1], "k--", lw=linewidth)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic for multi-class (One-Vs-All")
        plt.legend(loc="lower right")

    def plot_confusion_matrix(
        self, normalize=False, title="Confusion matrix", cmap=plt.cm.Reds
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Define the confusion matrix
        y_pred_class = argmax(self.y_pred, axis=1)
        cm = confusion_matrix(self.y_true, y_pred_class, labels=self.labels)

        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, newaxis]
            title = title + " normalized"

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = arange(len(self.labels))
        plt.xticks(tick_marks, self.labels, rotation=45)
        plt.yticks(tick_marks, self.labels)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    def print_report(self):
        from sklearn.metrics import classification_report

        y_pred_class = argmax(self.y_pred, axis=1)

        print("                   ________________________")
        print("                  |  Classification Report |")
        print("                   ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾")
        print(
            classification_report(
                self.y_true, y_pred_class, target_names=list(map(str, self.labels))
            )
        )
