from os import listdir
from os.path import isfile, join, exists
from params import parse_arguments, default_params

path_to_data = '../../'

def main(**params):
    params = dict(
        default_params,
        **params
    )

    print("Quick check to verify data is correctly located (by default parameters)")

    if not exists(join(path_to_data, params['data_path'])):
        print("Error: Could NOT locate data path")

    else:
        images_path = join(path_to_data, params['images_path'])

        images = [f for f in listdir(images_path) if isfile(join(images_path, f)) and f.endswith(".jpg")]
        assert len(images) == params['num_images']

        labels = params['labels']
        num_labels = params['num_labels']
        for i, lab in enumerate(labels):
            path_lab = join(join(path_to_data, params['gt_path']), lab)
            files = [f for f in listdir(path_lab) if isfile(join(path_lab, f)) and f.endswith(".tif")]
            assert len(files) == num_labels[i]

        print("Check DONE correctly")


if __name__ == '__main__':
    main(**vars(parse_arguments()))
