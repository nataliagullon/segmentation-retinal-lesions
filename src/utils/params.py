import argparse

default_params = {
    'data_path': 'data/',
    'images_path': 'data/images/',
    'gt_path': 'data/ground_truths/',
    'weights_path': 'model/',
    'labels': ['EX/', 'HE/', 'MA/', 'SE/'],
    'num_images': 54,
    'num_labels': [54, 53, 54, 26],
    'color_code_labels': [
    [0, 0, 0],      # 0 - Black     - Background
    [255, 0, 0],    # 1 - Red       - EX class
    [0, 255, 0],    # 2 - Green     - HE class
    [0, 0, 255],    # 3 - Blue      - MA class
    [255, 255, 0],  # 4 - Yellow    - SE class
    ],
    'seed': 8,
    'ratio_test': 0.1,  # ratio of dataset used in test
    'ratio_val': 0.1,   # ratio of dataset used in validation
    'patch_size': 400,
    'patch_size_test': 1600,
    'channels': 3,
    'thres_mask': 13,
    'n_classes': 5,
    'batch_size': 8,
    'lr': 1e-4
}


def parse_arguments():
    """
    Set arguments from the command line when running 'run.py'. Run with option '-h' or '--help' for
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-V', '--verbose', dest='verbose', action='store_true',
        help='provide additional details about the program', default=False
    )
    #parser.add_argument(
    #    '--split_by_phrase', action='store_true',
    #    help='blablabla', default=False
    #)
    parser.set_defaults(**default_params)

    return parser.parse_args()


def parse_arguments_pred():
    """
    Set arguments from the command line when running 'run.py'. Run with option '-h' or '--help' for
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-V', '--verbose', dest='verbose', action='store_true',
        help='provide additional details about the program', default=False
    )

    required_arg = parser.add_argument_group('required arguments')
    required_arg.add_argument(
        '-weights', dest='weights', help='name of the weights to use in the prediction', required=True
    )

    parser.set_defaults(**default_params)

    return parser.parse_args()


def parse_arguments_unet():
    """
    Set arguments from the command line when running 'run.py'. Run with option '-h' or '--help' for
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-V', '--verbose', dest='verbose', action='store_true',
        help='provide additional details about the program', default=False
    )

    required_arg = parser.add_argument_group('required arguments')
    required_arg.add_argument(
        '-weights', dest='weights', help='name of the weights that will be saved', required=True
    )

    parser.set_defaults(**default_params)

    return parser.parse_args()


def parse_arguments_gan():
    """
    Set arguments from the command line when running 'run.py'. Run with option '-h' or '--help' for
    information about parameters and its usage.
    :return: parser.parse_args()
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        argument_default=argparse.SUPPRESS
    )
    parser.add_argument(
        '-V', '--verbose', dest='verbose', action='store_true',
        help='provide additional details about the program', default=False
    )

    required_arg = parser.add_argument_group('required arguments')
    required_arg.add_argument(
        '-init_weights', dest='init_weights', help='initial weights to be loaded on the unet', required=True
    )
    required_arg.add_argument(
        '-weights', dest='weights', help='name of the weights that will be saved', required=True
    )

    parser.set_defaults(**default_params)

    return parser.parse_args()
