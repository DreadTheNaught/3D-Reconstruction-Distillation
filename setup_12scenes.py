import os
import zipfile
import urllib.request
import shutil
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to)


def mkdir(directory):
    """Checks whether the directory exists and creates it if necessary."""
    if not os.path.exists(directory):
        os.makedirs(directory)


# name of the folder where we download the original 12scenes dataset to
# we restructure the dataset by creating copies to that folder
src_folder = '12scenes_source'
# download the original 12 scenes dataset for calibration, poses and images
mkdir(src_folder)
os.chdir(src_folder)

datasets = ['office1', 'office2']
for ds in datasets:
    print(
        f"=== Downloading 12scenes Data: {ds} ===============================")
    url = f'http://graphics.stanford.edu/projects/reloc/data/{ds}.zip'
    zip_file = f'{ds}.zip'

    # Download the zip file with progress bar
    download_url(url, zip_file)

    # Unpack and delete zip file
    print(f"Extracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Get list of files for progress bar
        file_list = zip_ref.namelist()
        for file in tqdm(file_list, desc="Extracting files", unit="file"):
            zip_ref.extract(file)
    os.remove(zip_file)

    scenes = os.listdir(ds)
    for scene in scenes:
        data_folder = os.path.join(ds, scene, 'data')
        if not os.path.isdir(data_folder):
            # skip README files
            continue
        print(f"Processing files for 12scenes_{ds}_{scene}...")
        target_folder = os.path.join('..', f'12scenes_{ds}_{scene}')

        # create subfolders for training and test
        mkdir(os.path.join(target_folder, 'test', 'rgb'))
        mkdir(os.path.join(target_folder, 'test', 'poses'))
        mkdir(os.path.join(target_folder, 'test', 'calibration'))
        mkdir(os.path.join(target_folder, 'train', 'rgb'))
        mkdir(os.path.join(target_folder, 'train', 'poses'))
        mkdir(os.path.join(target_folder, 'train', 'calibration'))

        # read the train / test split - the first sequence is used for testing, everything else for training
        with open(os.path.join(ds, scene, 'split.txt'), 'r') as f:
            split = f.readlines()
        split = int(split[0].split()[1][8:-1])

        # read the calibration parameters, we use only the focallength
        with open(os.path.join(ds, scene, 'info.txt'), 'r') as f:
            focallength = f.readlines()
        focallength = focallength[7].split()
        focallength = (float(focallength[2]) + float(focallength[7])) / 2

        files = os.listdir(data_folder)
        images = [f for f in files if f.endswith('color.jpg')]
        images.sort()
        poses = [f for f in files if f.endswith('pose.txt')]
        poses.sort()

        def link_frame(i, variant):
            """ Creates copies of calibration, pose and image of frame i in either test or training. 
                Windows often requires admin privileges for symlinks, so we use copy instead. """
            # some image have invalid pose files, skip those
            valid = True
            with open(os.path.join(ds, scene, 'data', poses[i]), 'r') as f:
                pose = f.readlines()
                for line in pose:
                    if 'INF' in line:
                        valid = False

            if not valid:
                return False
            else:
                # copy pose and image
                src_img = os.path.join(data_folder, images[i])
                dst_img = os.path.join(
                    target_folder, variant, 'rgb', images[i])
                shutil.copy2(src_img, dst_img)

                src_pose = os.path.join(data_folder, poses[i])
                dst_pose = os.path.join(
                    target_folder, variant, 'poses', poses[i])
                shutil.copy2(src_pose, dst_pose)

                # create a calibration file
                calib_file = os.path.join(
                    target_folder, variant, 'calibration', f'frame-{str(i).zfill(6)}.calibration.txt')
                with open(calib_file, 'w') as f:
                    f.write(str(focallength))
                return True

        # frames up to split are test images
        print(f"Processing test frames (0-{split})...")
        skipped = 0
        for i in tqdm(range(split), desc="Test frames", unit="frame"):
            if not link_frame(i, 'test'):
                skipped += 1
        if skipped > 0:
            print(f"Skipped {skipped} test frames with corrupt poses.")

        # all remaining frames are training images
        print(f"Processing training frames ({split}-{len(images)-1})...")
        skipped = 0
        for i in tqdm(range(split, len(images)), desc="Training frames", unit="frame"):
            if not link_frame(i, 'train'):
                skipped += 1
        if skipped > 0:
            print(f"Skipped {skipped} training frames with corrupt poses.")

# Return to the original directory
os.chdir('..')
print("\nProcessing complete! Dataset organized successfully.")
