#!/usr/bin/env python
#
# cni_challenge ds ChRIS plugin app
#
# (c) 2016-2019 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import os
import sys
sys.path.append(os.path.dirname(__file__))

# import the Chris app superclass
from chrisapp.base import ChrisApp
# Import a pythton object of the classifier that does the prediction
from classification import predict_diagnosis
import os


Gstr_title = """

            _  _____  _____  __   _____      _           _ _
           (_)/ __  \|  _  |/  | |  _  |    | |         | | |
  ___ _ __  _ `' / /'| |/' |`| | | |_| | ___| |__   __ _| | | ___ _ __   __ _  ___
 / __| '_ \| |  / /  |  /| | | | \____ |/ __| '_ \ / _` | | |/ _ \ '_ \ / _` |/ _ \\
| (__| | | | |./ /___\ |_/ /_| |_.___/ / (__| | | | (_| | | |  __/ | | | (_| |  __/
 \___|_| |_|_|\_____/ \___/ \___/\____/ \___|_| |_|\__,_|_|_|\___|_| |_|\__, |\___|
                                    ______                               __/ |
                                   |______|                             |___/

"""

Gstr_synopsis = """

    NAME

       cni_challenge.py

    SYNOPSIS

        python cni_challenge.py                                         \\
            [-h] [--help]                                               \\
            [--json]                                                    \\
            [--man]                                                     \\
            [--meta]                                                    \\
            [--savejson <DIR>]                                          \\
            [-v <level>] [--verbosity <level>]                          \\
            [--version]                                                 \\
            <inputDir>                                                  \\
            <outputDir>                                                 \\
            [--rot <matrix_file.txt>]                                   \\

    BRIEF EXAMPLE

        * Bare bones execution of a python example to read in a vector file, perform a matrix rotation, and output the
          new vectors in a text file.

            mkdir inputdir outputdir && chmod 777 outputdir
            python cni_challenge.py inputdir outputdir  --run_option python --rot rotation_matrices.txt

            N.B. Required files (rotation_matrices.txt and vectors.txt) should be in 'inputdir' as provided in cni_challenge
            github repository.

            Output will be outputdir/classification.txt.

    DESCRIPTION

        `cni_challenge.py` has been created for MICCAI CNI 2019 Challenge
        http://www.brainconnectivity.net.

        Solutions should be incorporated into this package and a container created through Docker.
        Submission to the Challenge will be a link to the Docker container.

        `cni_challenge.py` contains currently contains a running python example.

    ARGS

        <inputDir>
        Mandatory. A directory which contains all necessary input files.

        <outputDir>
        Mandatory. A directory where output will be saved to. Must be universally writable to.

        [-h] [--help]
        If specified, show help message and exit.

        [--json]
        If specified, show json representation of app and exit.

        [--man]
        If specified, print (this) man page and exit.

        [--meta]
        If specified, print plugin meta data and exit.

        [--savejson <DIR>]
        If specified, save json representation file to DIR and exit.

        [-v <level>] [--verbosity <level>]
        Verbosity level for app. Not used currently.

        [--version]
        If specified, print version number and exit.

"""


class Cni_challenge(ChrisApp):
    """
    A bare bones app created for MICCAI CNI 2019 Challenge.
    Challengers are to use this app to create a Docker container of their solution in order to submit it.
    """
    AUTHORS                 = 'AWC (aiwern.chung@childrens.harvard.edu)'
    SELFPATH                = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC                = os.path.basename(__file__)
    EXECSHELL               = 'python3'
    TITLE                   = 'A ChRIS plugin for the CNI 2019 Challenge'
    CATEGORY                = ''
    TYPE                    = 'ds'
    DESCRIPTION             = 'An app for contestants to create a Docker container of a solution to the CNI 2019 Challenge.\n' \
                              'For help see: http://www.brainconnectivity.net/challenge_subm.html'
    DOCUMENTATION           = 'http://wiki'
    VERSION                 = '0.1'
    ICON                    = '' # url of an icon image
    LICENSE                 = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS   = 1  # Override with integer value
    MAX_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT           = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT        = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT           = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT           = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Use this dictionary structure to provide key-value output descriptive information
    # that may be useful for the next downstream plugin. For example:
    #
    # {
    #   "finalOutputFile":  "final/file.out",
    #   "viewer":           "genericTextViewer",
    # }
    #
    # The above dictionary is saved when plugin is called with a ``--saveoutputmeta``
    # flag. Note also that all file paths are relative to the system specified
    # output directory.
    OUTPUT_META_DICT = {}


    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        Use self.add_argument to specify a new app argument.
        """

        # To pass in a string
        print("No additional paramas are requested")


    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """

        print(Gstr_title)
        print('Version: %s' % self.get_version())

        # ===============================================
        # Initialising variables
        # Input and output files must be in 'inputdir' and 'outputdir', respectively.
        # ===============================================

        classifier = 'classifier/classifier.joblib'                        # the classifier to be used in prediction
        input_dir = '%s/' % (options.inputdir)                             # input directory containing input variables
        out_dir= '%s/' % (options.outputdir)                               # putput directory containing the output of the classification
        evaluate_classification='evaluation/classification_metrics.py'     # path to the python script to evaluate classification
        classification_file=out_dir+'/classification.txt'                  # path to the txt file containig the predictions
        goundtruth_file=out_dir+'/goundtruth.txt'                          # path to the txt file containig the real data
        output_file=out_dir+'/scores.txt'                                  # path to the txt file containig the perfomance of the classifier

        # ===============================================
        # Call code
        # ===============================================
        print("\n")
        print("\tCalling python code to perform classification on aal atlas data ...")
        predict_diagnosis(input_dir, out_dir, classifier)
        print("\tCalling python code to evaluate the perfomance of the classifier ...")
        os.system("python3 "+evaluate_classification+" -p "+classification_file+" -g "+goundtruth_file+" -o "+output_file)
        print ("\tOutput will be in %s" % out_dir)
        print("====================================================================================")

    def show_man_page(self):
        """
        Print the app's man page.
        """
        print(Gstr_synopsis)


# ENTRYPOINT
if __name__ == "__main__":
    chris_app = Cni_challenge()
    chris_app.launch()
