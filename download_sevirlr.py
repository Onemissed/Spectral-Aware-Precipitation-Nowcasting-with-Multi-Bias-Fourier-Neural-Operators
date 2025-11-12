import os

default_dataset_sevirlr_dir = os.path.join('data', "sevirlr")

def download_SEVIRLR(save_dir=None):
    r"""
    Downloaded dataset is saved in save_dir/sevirlr
    """
    if save_dir is None:
        save_dir = default_dataset_sevirlr_dir
    else:
        save_dir = os.path.join(save_dir, "sevirlr")
    if os.path.exists(save_dir):
        raise FileExistsError(f"Path to save SEVIR-LR dataset {save_dir} already exists!")
    else:
        os.makedirs(save_dir)
        os.system(f"wget https://deep-earth.s3.amazonaws.com/datasets/sevir_lr.zip "
                  f"-P {os.path.abspath(save_dir)}")
        os.system(f"unzip {os.path.join(save_dir, 'sevir_lr.zip')} "
                  f"-d {save_dir}")
        os.system(f"mv {os.path.join(save_dir, 'sevir_lr', '*')} "
                  f"{save_dir}\n"
                  f"rm -rf {os.path.join(save_dir, 'sevir_lr')}")

if __name__ == '__main__':
    download_SEVIRLR()
