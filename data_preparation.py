# from google.colab import drive

# from pathlib import Path
from typing import List, Dict

# import matplotlib.pyplot as plt
# import pandas as pd
import json
import cv2
import collections
import os

import numpy as np
import pandas as pd
import argparse
import subprocess

try:
    import skvideo.io
except ImportError:
    subprocess.check_call(['pip', 'install', 'sk-video'])
    import skvideo.io

try:
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug import parameters as iap
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
except ImportError:
    import subprocess

    subprocess.check_call(['pip', 'install', 'imgaug'])
    import imgaug as ia
    import imgaug.augmenters as iaa
    from imgaug import parameters as iap
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# from PIL import Image

CONVERTED_H = 480
CONVERTED_W = 640
ORIGINAL_H = 1520
ORIGINAL_W = 2704
LAST_FRAME_ORIG = 31785


def save_frames_to_file(video_path: str, frames_dir: str, points_ids: List[str], frames: List[str],
                        fish_codes: List[str], fish_stages: List[str], bg: bool = False):
    vid_cap = skvideo.io.vreader(video_path)
    count = 0
    try:
        for point_id, frame_num, fish_code, fish_stage in zip(points_ids, frames, fish_codes, fish_stages):
            point_dirname = f'{fish_stage}_{fish_code}' if not bg else 'background'
            point_dir = os.path.join(frames_dir, point_dirname)
            os.makedirs(point_dir, exist_ok=True)
            point_path = os.path.join(point_dir, point_id + '.jpg')
            if os.path.exists(point_path):
                return
            print('Saving:', point_id)
            for frame in vid_cap:
                if count == frame_num:
                    writer = skvideo.io.FFmpegWriter(point_path)
                    writer.writeFrame(frame)
                    writer.close()
                    break
                count += 1
    except ValueError:
        pass


def save_background(video_name: str, video_path: str, frames_dir: str, start_frame: int = 10000,
                    step: int = 100, max_num_frames: int = 10000):
    # TODO - try to run
    frame_numbers = list(range(start_frame, LAST_FRAME_ORIG, step))[:max_num_frames]
    point_ids = [video_name + '_' + str(frame) for frame in frame_numbers]
    sp_codes = stages = [''] * len(point_ids)
    save_frames_to_file(video_path=video_path, frames_dir=frames_dir,
                        points_ids=point_ids, frames=frame_numbers, fish_codes=sp_codes, fish_stages=stages,
                        bg=True)


def create_frames(video_dir: str, frames_dir: str, tags: Dict[str, Dict], converted: bool = True):
    for experiment_name, experiment_dict in tags.items():
        if not os.path.exists(os.path.join(video_dir, experiment_name)):
            continue
        print("Experiment name:", experiment_name)
        for video_key, video_dict in experiment_dict.items():
            video_name = os.path.splitext(video_key)[0] + '-converted.mp4' if converted else video_key
            video_path = os.path.join(video_dir, experiment_name, video_name)
            if not os.path.exists(video_path):
                continue
            point_ids = []
            frames_num = []
            sp_codes = []
            stages = []
            # point_data: 'frame', 'time', 'family', 'genus', 'species', 'spcode', 'stage', 'activity',
            # 'length', 'xh', 'yh', 'xt', 'yt'
            for point_id, point_data in video_dict.items():
                mul = 0.5 if experiment_name in \
                             ['israchz091121A', 'israchz091121B', 'israchz091121C', 'isrtche081121B'] else 1
                frames_num.append(int(point_data['frame'] * mul))
                point_ids.append(point_id)
                sp_codes.append(point_data['spcode'])
                stages.append(point_data['stage'])
            save_frames_to_file(video_path=video_path, frames_dir=frames_dir,
                                points_ids=point_ids, frames=frames_num, fish_codes=sp_codes, fish_stages=stages)


def create_bg_frames(video_dir: str, frames_dir: str):
    experiments = [
        'isrrosh071020A ', 'isrrosh081020B', 'isrrosh030221C', 'israchz260521A', 'isrtche081121A',
        'israchz040221C', 'isrrosh071020B']
    bg_videos = ['lGOPR', 'rGOPR', 'raissrosh', 'rbissrosh', 'raisrsdot201119B', 'raisrnaha141119A', 'rbisrsdot201119B',
                 'rbisrnaha141119A', 'raisrnaha121119A', 'rbisrnaha121119A']
    os.makedirs(os.path.join(frames_dir, 'background'), exist_ok=True)
    for experiment_name in experiments:
        experiment_path = os.path.join(video_dir, experiment_name)
        if not os.path.exists(experiment_path):
            continue
        print("Experiment name:", experiment_name)
        for video_name in os.listdir(experiment_path):
            if not any([video_name.startswith(v) for v in bg_videos]):
                continue
            video_path = os.path.join(video_dir, experiment_name, video_name)
            video_basename = os.path.splitext(video_name)[0]
            save_background(
                video_basename, video_path, frames_dir=frames_dir, start_frame=12000, step=100, max_num_frames=3)


def create_tags_json(data_dir: str):
    metadata = pd.read_csv(
        os.path.join(data_dir, '2022-1107_med_campaign_stereoBRUVs_Lengths.txt'), sep='\t', header=0).fillna(0)
    coords = pd.read_csv(
        os.path.join(data_dir, '2022-1107_med_campaign_stereoBRUVs_ImagePtPair.txt'), sep='\t', header=0).fillna(0)

    metadata['Code'] = metadata['Code'].astype(int).astype(str)

    exps_dict = {}
    experiment_groups = metadata.groupby('OpCode')
    for exp_name, expdata in experiment_groups:
        exps_dict[exp_name] = collections.defaultdict(dict)
        l_videos_groups = expdata.groupby('FilenameLeft')
        for l_name, data in l_videos_groups:
            video_name_by_direction = {'L': l_name, 'R': data.FilenameRight.iloc[0]}
            for line in data.itertuples():
                # unpolished datapoint in 'isrrosh141119B':
                directions = ['L'] if 'isrrosh141119B' in l_name else ['L', 'R']
                for direction in directions:
                    if int(line.Code) == 0:
                        continue
                    points = dict()
                    points['frame'] = line.FrameLeft if direction == 'L' else line.FrameRight
                    points['family'] = line.Family
                    points['genus'] = line.Genus
                    points['species'] = line.Species
                    points['spcode'] = str(int(line.Code))
                    points['stage'] = line.Stage

                    points_df = coords[coords['OpCode'] == exp_name][coords['ImagePtPair'] == line.ImagePtPair]
                    points['xh'] = points_df.iloc[0][direction + 'x']
                    points['yh'] = points_df.iloc[0][direction + 'y']
                    points['xt'] = points_df.iloc[1][direction + 'x']
                    points['yt'] = points_df.iloc[1][direction + 'y']

                    point_id = exp_name + '_' + str(line.ImagePtPair) + '_' + direction.lower()
                    exps_dict[exp_name][video_name_by_direction[direction]][point_id] = points

    with open(os.path.join(data_dir, 'labels.json'), 'w+') as f_:
        json.dump(exps_dict, f_, sort_keys=True, indent=4)


def convert_to_bb(xh: float, yh: float, xt: float, yt: float):
    xh = xh * CONVERTED_W / ORIGINAL_W
    yh = yh * CONVERTED_H / ORIGINAL_H
    xt = xt * CONVERTED_W / ORIGINAL_W
    yt = yt * CONVERTED_H / ORIGINAL_H
    length = np.sqrt((xt - xh) ** 2 + (yt - yh) ** 2)
    x2 = max(xh, xt)
    x1 = min(xh, xt)
    y2 = max(yh, yt)
    y1 = min(yh, yt)
    if (x2 - x1) > (y2 - y1):
        yc = (y1 + y2) / 2
        y1 = max(0, yc - length / 2)
        y2 = min(ORIGINAL_H, yc + length / 2)
    else:
        xc = (x1 + x2) / 2
        x1 = max(xc - length / 2, 0)
        x2 = min(xc + length / 2, ORIGINAL_W)
    return x1, y1, x2, y2


def build_data_df(video_dir: str, frames_dir: str, tags: Dict[str, Dict], mode: int = 0) -> pd.DataFrame:
    """mode: int [0,1,2]
    0: samples + background
    1: only samples
    2: only background"""

    header = ['image_id', 'x1', 'y1', 'x2', 'y2', 'frame_path', 'label', 'species']
    data_points = {h: [] for h in header}

    def add_to_samples_dict(image_id_, frame_path_, x1_, x2_, y1_, y2_, label_, sp_):
        data_points['image_id'].append(image_id_)
        data_points['frame_path'].append(frame_path_)
        data_points['x1'].append(x1_)
        data_points['x2'].append(x2_)
        data_points['y1'].append(y1_)
        data_points['y2'].append(y2_)
        data_points['label'].append(label_)
        data_points['species'].append(sp_)

    # Add sample images
    if mode != 2:
        for sample, sample_dict in tags.items():
            if not os.path.isdir(os.path.join(video_dir, sample)):
                continue
            for v_name, v_dict in sample_dict.items():
                for f_name, f_dict in v_dict.items():
                    c = f"{f_dict['stage']}_{f_dict['spcode']}"
                    sp = f"{f_dict['genus']}_{f_dict['species'].replace(' ', '_')}"
                    species_dir = os.path.join(frames_dir, c)
                    if not os.path.isfile(os.path.join(species_dir, f_name + '.jpg')):
                        continue
                    x1, y1, x2, y2 = convert_to_bb(f_dict['xh'], f_dict['yh'], f_dict['xt'], f_dict['yt'])
                    add_to_samples_dict('_'.join([sample, os.path.splitext(v_name)[0], str(f_dict['frame'])]),
                                        os.path.join(species_dir, f_name + '.jpg'),
                                        x1, x2, y1, y2, c, sp)

    # Add background images
    bg_path = os.path.join(frames_dir, 'background')
    if mode != 1:
        for bg_frame_filename in os.listdir(bg_path):
            add_to_samples_dict('BG_' + bg_frame_filename.split('.')[0],
                                os.path.join(bg_path, bg_frame_filename),
                                0, CONVERTED_W, 0, CONVERTED_H, 'background', 'background')

    filename = 'data.csv' if mode == 1 else 'background.csv' if mode == 2 else 'data_and_background.csv'
    df = pd.DataFrame(data_points, columns=header)
    df.to_csv(os.path.join(frames_dir, filename), index=False)

    train, validate, test = np.split(df['image_id'].sample(frac=1, random_state=1234),
                                     [int(.8 * len(df)), int(.9 * len(df))])
    df[df['image_id'].isin(list(train))].to_csv(os.path.join(frames_dir, 'train.csv'), index=False)
    df[df['image_id'].isin(list(validate))].to_csv(os.path.join(frames_dir, 'validate.csv'), index=False)
    df[df['image_id'].isin(list(test))].to_csv(os.path.join(frames_dir, 'test.csv'), index=False)
    return df


def augment_image(images, bboxes):
    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        iaa.LinearContrast((0.5, 2.0), per_channel=0.5)
    ])
    bbs = BoundingBoxesOnImage([BoundingBox(*list(b)) for b in list(bboxes)], shape=images.shape)
    images_aug, bbs_aug = seq(image=images, bounding_boxes=bbs)
    bbs_aug = np.array([[b.x1, b.y1, b.x2, b.y2] for b in bbs_aug])
    return images_aug, bbs_aug


def add_augmentations(frames_dir: str, num_augs=10):
    df = pd.read_csv(os.path.join(frames_dir, 'train.csv'))
    label_counter = collections.Counter(list(df['label']))
    rare_labels = [l for l, c in label_counter.items() if c < 10]
    filtered_df = df[df['label'].isin(rare_labels)]
    image_ids = filtered_df['image_id'].unique()
    print('Creating augmentations for:')
    for im_id in image_ids:
        print('Image ID -', im_id)
        image_df = df[df['image_id'] == im_id]
        frame_path = list(image_df['frame_path'])[0]
        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = np.zeros((image_df.shape[0], 4))
        boxes[:, 0:4] = image_df[['x1', 'y1', 'x2', 'y2']].values
        augs_df = pd.DataFrame(columns=image_df.columns)
        for i in range(num_augs):
            aug_path = os.path.splitext(frame_path)[0] + '_' + str(i) + '.jpg'
            image_aug, bbs_aug = augment_image(images=image, bboxes=boxes)
            cv2.imwrite(aug_path, image_aug)
            aug_df = pd.DataFrame(bbs_aug, columns=['x1', 'y1', 'x2', 'y2'])
            aug_df['label'] = image_df['label'].iloc[0] if len(image_df) == 1 else list(image_df['label'])
            aug_df['species'] = image_df['species'].iloc[0] if len(image_df) == 1 else list(image_df['species'])
            # aug_df['origin'] = image_df['origin'].iloc[0] if len(image_df) == 1 else list(image_df['origin'])
            aug_df['image_id'] = image_df['image_id'].iloc[0] + '_' + str(i)
            aug_df['frame_path'] = aug_path
            augs_df = pd.concat([augs_df, aug_df])
        df = pd.concat([df, augs_df])
    df.to_csv(os.path.join(frames_dir, 'augs_data.csv'), index=False)
    return df


def prepare_data(data_dir: str = 'raw_data', frames_dir: str = 'frames', is_converted: bool = False,
                 data_types: int = 0, create_augs: bool = False, num_augs: int = 10, to_create_frames: bool = False,
                 to_create_bg_frames: bool = False):
    create_tags_json(data_dir=data_dir)

    with open(os.path.join(data_dir, 'labels.json')) as f:
        tags = json.load(f)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    if to_create_frames:
        create_frames(video_dir=data_dir, frames_dir=frames_dir, tags=tags, converted=is_converted)
    if to_create_bg_frames:
        create_bg_frames(video_dir=data_dir, frames_dir=frames_dir)

    build_data_df(video_dir=data_dir, frames_dir=frames_dir, tags=tags, mode=data_types)

    if create_augs:
        add_augmentations(frames_dir=frames_dir, num_augs=num_augs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data-dir', help='Raw Data Directory Name', required=True)
    parser.add_argument('-f', '--frames-dir', help='Frames Directory Name', default='frames')
    parser.add_argument('-l', '--labels-dir', help='Labels Directory Name', default='labels')
    parser.add_argument('--create-frames', help='Extract the relevant frames from the videos', action='store_true')
    parser.add_argument('--create-bg-frames', help='Extract the background frames from the videos', action='store_true')
    parser.add_argument('--create-augs', help='Create augmentations for the frames', action='store_true')
    parser.add_argument('--is-converted', help='Using the converted videos', action='store_true')
    parser.add_argument('--data-types', help='Choose between using only the samples (2), only the background (1) '
                                             'or using both (0), default is 0.', default=0, type=int, choices=[0, 1, 2])
    parser.add_argument('--num-augs', help='Number of augmentations for each relevant frame', default=10, type=int)

    # Read arguments from command line
    args = parser.parse_args()

    prepare_data(data_dir=args.data_dir, frames_dir=args.frames_dir, is_converted=args.is_converted,
                 data_types=args.data_types, create_augs=args.create_augs, num_augs=args.num_augs,
                 to_create_frames=args.create_frames, to_create_bg_frames=args.create_bg_frames)


