# -*- coding: utf-8 -*-
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from collections import Counter
from pathlib import Path
import subprocess
import importlib
import math
import sys
import glob
import json
import pickle
import re
import warnings

from sklearn.datasets.base import Bunch
from skimage.draw import polygon
import skimage.transform
import shapely.wkt
from shapely.geometry import MultiPolygon, Polygon
import pandas as pd
import numpy as np
import tables as tb
import scipy
import rasterio
import rasterio.features
import tqdm
import cv2
import gdal
import click
import skimage.draw
import shapely.wkt
import shapely.ops
import shapely.geometry
import fiona
import affine
from keras.models import Model
from keras.engine.topology import merge as merge_l
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout,
    Activation, BatchNormalization)
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K


MODEL_NAME = 'v16'
ORIGINAL_SIZE = 650
INPUT_SIZE = 256
STRIDE_SZ = 197

LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
BASE_DIR = "/data/train"  # train data
BASE_TEST_DIR = "/data/test"  # test data
WORKING_DIR = "/data/working"
IMAGE_DIR = "/data/working/images/{}".format('v16')
V12_IMAGE_DIR = "/data/working/images/{}".format('v12')  # for mask and mul
V5_IMAGE_DIR = "/data/working/images/{}".format('v5')
MODEL_DIR = "/data/working/models/{}".format(MODEL_NAME)
FN_SOLUTION_CSV = "/data/output/{}.csv".format(MODEL_NAME)

# ---------------------------------------------------------
# Parameters
MIN_POLYGON_AREA = 30

# ---------------------------------------------------------
# Input files
FMT_TRAIN_SUMMARY_PATH = str(
    Path(BASE_DIR) /
    Path("{prefix:s}_Train/") /
    Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
FMT_TRAIN_RGB_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TEST_RGB_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TRAIN_MSPEC_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
FMT_TEST_MSPEC_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))

# ---------------------------------------------------------
# Preprocessing result
FMT_RGB_BANDCUT_TH_PATH = V12_IMAGE_DIR + "/rgb_bandcut{}.csv"
FMT_MUL_BANDCUT_TH_PATH = V12_IMAGE_DIR + "/mul_bandcut{}.csv"

# ---------------------------------------------------------
# Image list, Image container and mask container
FMT_VALTRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_TRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
FMT_TEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"

# Mask
FMT_VALTRAIN_MASK_STORE = V12_IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTEST_MASK_STORE = V12_IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_TRAIN_MASK_STORE = V12_IMAGE_DIR + "/train_{}_mask.h5"

# MUL
FMT_VALTRAIN_MUL_STORE = V12_IMAGE_DIR + "/valtrain_{}_mul.h5"
FMT_VALTEST_MUL_STORE = V12_IMAGE_DIR + "/valtest_{}_mul.h5"
FMT_TRAIN_MUL_STORE = V12_IMAGE_DIR + "/train_{}_mul.h5"
FMT_TEST_MUL_STORE = V12_IMAGE_DIR + "/test_{}_mul.h5"
FMT_MULMEAN = V12_IMAGE_DIR + "/{}_mulmean.h5"

# OSM
FMT_VALTRAIN_OSM_STORE = IMAGE_DIR + "/valtrain_{}_osm.h5"
FMT_VALTEST_OSM_STORE = IMAGE_DIR + "/valtest_{}_osm.h5"
FMT_TRAIN_OSM_STORE = IMAGE_DIR + "/train_{}_osm.h5"
FMT_TEST_OSM_STORE = IMAGE_DIR + "/test_{}_osm.h5"
FMT_OSM_MEAN = IMAGE_DIR + "/{}_osmmean.h5"

# ---------------------------------------------------------
# Model files
FMT_VALMODEL_PATH = MODEL_DIR + "/{}_val_weights.h5"
FMT_FULLMODEL_PATH = MODEL_DIR + "/{}_full_weights.h5"
FMT_VALMODEL_HIST = MODEL_DIR + "/{}_val_hist.csv"
FMT_VALMODEL_EVALHIST = MODEL_DIR + "/{}_val_evalhist.csv"
FMT_VALMODEL_EVALTHHIST = MODEL_DIR + "/{}_val_evalhist_th.csv"

# ---------------------------------------------------------
# Prediction & polygon result
FMT_TESTPRED_PATH = MODEL_DIR + "/{}_pred.h5"
FMT_VALTESTPRED_PATH = MODEL_DIR + "/{}_eval_pred.h5"
FMT_VALTESTPOLY_PATH = MODEL_DIR + "/{}_eval_poly.csv"
FMT_VALTESTTRUTH_PATH = MODEL_DIR + "/{}_eval_poly_truth.csv"
FMT_VALTESTPOLY_OVALL_PATH = MODEL_DIR + "/eval_poly.csv"
FMT_VALTESTTRUTH_OVALL_PATH = MODEL_DIR + "/eval_poly_truth.csv"
FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"

# ---------------------------------------------------------
# Model related files (others)
FMT_VALMODEL_LAST_PATH = MODEL_DIR + "/{}_val_weights_last.h5"
FMT_FULLMODEL_LAST_PATH = MODEL_DIR + "/{}_full_weights_last.h5"

# OSM dataset (Extracted from https://mapzen.com/data/metro-extracts/)
FMT_OSMSHAPEFILE = "/root/osmdata/{name:}/{name:}_{layer:}.shp"
FMT_SERIALIZED_OSMDATA = WORKING_DIR + "/osm_{}_subset.pkl"
LAYER_NAMES = [
    'buildings',
    'landusages',
    'roads',
    'waterareas',
]

# ---------------------------------------------------------
# warnins and logging
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
fh_handler = FileHandler(".{}.log".format(MODEL_NAME))
fh_handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
logger = getLogger(__name__)
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)
    logger.addHandler(fh_handler)


# Fix seed for reproducibility
np.random.seed(1145141919)


def directory_name_to_area_id(datapath):
    """
    Directory name to AOI number

    Usage:

        >>> directory_name_to_area_id("/data/test/AOI_2_Vegas")
        2
    """
    dir_name = Path(datapath).name
    if dir_name.startswith('AOI_2_Vegas'):
        return 2
    elif dir_name.startswith('AOI_3_Paris'):
        return 3
    elif dir_name.startswith('AOI_4_Shanghai'):
        return 4
    elif dir_name.startswith('AOI_5_Khartoum'):
        return 5
    else:
        raise RuntimeError("Unsupported city id is given.")


def _remove_interiors(line):
    if "), (" in line:
        line_prefix = line.split('), (')[0]
        line_terminate = line.split('))",')[-1]
        line = (
            line_prefix +
            '))",' +
            line_terminate
        )
    return line


def _calc_fscore_per_aoi(area_id):
    prefix = area_id_to_prefix(area_id)
    truth_file = FMT_VALTESTTRUTH_PATH.format(prefix)
    poly_file = FMT_VALTESTPOLY_PATH.format(prefix)

    cmd = [
        'java',
        '-jar',
        '/root/visualizer-2.0/visualizer.jar',
        '-truth',
        truth_file,
        '-solution',
        poly_file,
        '-no-gui',
        '-band-triplets',
        '/root/visualizer-2.0/data/band-triplets.txt',
        '-image-dir',
        'pass',
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_data, stderr_data = proc.communicate()
    lines = [line for line in stdout_data.decode('utf8').split('\n')[-10:]]

    """
Overall F-score : 0.85029

AOI_2_Vegas:
  TP       : 27827
  FP       : 4999
  FN       : 4800
  Precision: 0.847712
  Recall   : 0.852883
  F-score  : 0.85029
    """
    if stdout_data.decode('utf8').strip().endswith("Overall F-score : 0"):
        overall_fscore = 0
        tp = 0
        fp = 0
        fn = 0
        precision = 0
        recall = 0
        fscore = 0

    elif len(lines) > 0 and lines[0].startswith("Overall F-score : "):
        assert lines[0].startswith("Overall F-score : ")
        assert lines[2].startswith("AOI_")
        assert lines[3].strip().startswith("TP")
        assert lines[4].strip().startswith("FP")
        assert lines[5].strip().startswith("FN")
        assert lines[6].strip().startswith("Precision")
        assert lines[7].strip().startswith("Recall")
        assert lines[8].strip().startswith("F-score")

        overall_fscore = float(re.findall("([\d\.]+)", lines[0])[0])
        tp = int(re.findall("(\d+)", lines[3])[0])
        fp = int(re.findall("(\d+)", lines[4])[0])
        fn = int(re.findall("(\d+)", lines[5])[0])
        precision = float(re.findall("([\d\.]+)", lines[6])[0])
        recall = float(re.findall("([\d\.]+)", lines[7])[0])
        fscore = float(re.findall("([\d\.]+)", lines[8])[0])
    else:
        logger.warn("Unexpected data >>> " + stdout_data.decode('utf8'))
        raise RuntimeError("Unsupported format")

    return {
        'overall_fscore': overall_fscore,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'fscore': fscore,
    }


def prefix_to_area_id(prefix):
    area_dict = {
        'AOI_2_Vegas': 2,
        'AOI_3_Paris': 3,
        'AOI_4_Shanghai': 4,
        'AOI_5_Khartoum': 5,
    }
    return area_dict[area_id]


def area_id_to_prefix(area_id):
    area_dict = {
        2: 'AOI_2_Vegas',
        3: 'AOI_3_Paris',
        4: 'AOI_4_Shanghai',
        5: 'AOI_5_Khartoum',
    }
    return area_dict[area_id]


def area_id_to_osmprefix(area_id):
    area_id_to_osmprefix_dict = {
        2: 'las-vegas_nevada_osm',
        3: 'paris_france_osm',
        4: 'shanghai_china_osm',
        5: 'ex_s2cCo6gpCXAvihWVygCAfSjNVksnQ_osm',
    }
    return area_id_to_osmprefix_dict[area_id]


# ---------------------------------------------------------
# main


def _get_model_parameter(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_hist = FMT_VALMODEL_EVALTHHIST.format(prefix)
    best_row = pd.read_csv(fn_hist).sort_values(
        by='fscore',
        ascending=False,
    ).iloc[0]

    param = dict(
        fn_epoch=int(best_row['zero_base_epoch']),
        min_poly_area=int(best_row['min_area_th']),
    )
    return param


def _internal_test_predict_best_param(area_id,
                                      save_pred=True):
    prefix = area_id_to_prefix(area_id)
    param = _get_model_parameter(area_id)
    epoch = param['fn_epoch']
    min_th = param['min_poly_area']

    # Prediction phase
    logger.info("Prediction phase: {}".format(prefix))

    dict_n_osm_layers = {
        2: 4,
        3: 5,
        4: 4,
        5: 4,
    }
    osm_layers = dict_n_osm_layers[area_id]
    n_input_layers = 8 + osm_layers

    X_mean = get_mul_mean_image(area_id)
    X_osm_mean = np.zeros((
        osm_layers,
        INPUT_SIZE,
        INPUT_SIZE,
    ))
    X_mean = np.vstack([X_mean, X_osm_mean])

    # Load model weights
    # Predict and Save prediction result
    fn = FMT_TESTPRED_PATH.format(prefix)
    fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
    fn_model = fn_model.format(epoch=epoch)
    model = get_unet(input_layers=n_input_layers)
    model.load_weights(fn_model)

    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    y_pred = model.predict_generator(
        generate_test_batch(
            area_id,
            batch_size=64,
            immean=X_mean,
            enable_tqdm=True,
        ),
        val_samples=len(df_test) * 9,
    )
    del model

    # Save prediction result
    if save_pred:
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(y_pred.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
                                 filters=filters)
            ds[:] = y_pred

    return y_pred


def _internal_test(area_id, enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    y_pred = _internal_test_predict_best_param(area_id, save_pred=False)

    # Postprocessing phase
    logger.info("Postprocessing phase")
    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    fn_out = FMT_TESTPOLY_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")

        test_image_list = df_test.index.tolist()
        for idx, image_id in tqdm.tqdm(enumerate(test_image_list)):
            pred_values = np.zeros((650, 650))
            pred_count = np.zeros((650, 650))
            for slice_pos in range(9):
                slice_idx = idx * 9 + slice_pos

                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = STRIDE_SZ * pos_i
                y0 = STRIDE_SZ * pos_j
                pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
                    y_pred[slice_idx][0]
                )
                pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1
            pred_values = pred_values / pred_count

            df_poly = mask_to_poly(pred_values, min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))


def _internal_validate_predict_best_param(area_id,
                                          enable_tqdm=False):
    param = _get_model_parameter(area_id)
    epoch = param['fn_epoch']
    y_pred = _internal_validate_predict(
        area_id,
        epoch=epoch,
        save_pred=False,
        enable_tqdm=enable_tqdm)

    return y_pred


def _internal_validate_predict(area_id,
                               epoch=3,
                               save_pred=True,
                               enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)

    dict_n_osm_layers = {
        2: 4,
        3: 5,
        4: 4,
        5: 4,
    }
    osm_layers = dict_n_osm_layers[area_id]
    n_input_layers = 8 + osm_layers

    # Image Mean
    X_mean = get_mul_mean_image(area_id)
    X_osm_mean = np.zeros((
        osm_layers,
        INPUT_SIZE,
        INPUT_SIZE,
    ))
    X_mean = np.vstack([X_mean, X_osm_mean])

    # Load model weights
    # Predict and Save prediction result
    fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
    fn_model = fn_model.format(epoch=epoch)
    model = get_unet(input_layers=n_input_layers)
    model.load_weights(fn_model)

    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    y_pred = model.predict_generator(
        generate_valtest_batch(
            area_id,
            batch_size=32,
            immean=X_mean,
            enable_tqdm=enable_tqdm,
        ),
        val_samples=len(df_test) * 9,
    )
    del model

    # Save prediction result
    if save_pred:
        fn = FMT_VALTESTPRED_PATH.format(prefix)
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(y_pred.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(
                f.root,
                'pred',
                atom,
                y_pred.shape,
                filters=filters,
            )
            ds[:] = y_pred

    return y_pred


def _internal_validate_fscore_wo_pred_file(area_id,
                                           epoch=3,
                                           min_th=MIN_POLYGON_AREA,
                                           enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)

    # ------------------------
    # Prediction phase
    logger.info("Prediction phase")
    y_pred = _internal_validate_predict(
        area_id,
        save_pred=False,
        epoch=epoch,
        enable_tqdm=enable_tqdm)

    # ------------------------
    # Postprocessing phase
    logger.info("Postprocessing phase")
    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        test_list = df_test.index.tolist()
        iterator = enumerate(test_list)

        for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
            pred_values = np.zeros((650, 650))
            pred_count = np.zeros((650, 650))
            for slice_pos in range(9):
                slice_idx = idx * 9 + slice_pos

                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = STRIDE_SZ * pos_i
                y0 = STRIDE_SZ * pos_j
                pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
                    y_pred[slice_idx][0]
                )
                pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1
            pred_values = pred_values / pred_count

            df_poly = mask_to_poly(pred_values, min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))

    # ------------------------
    # Validation solution file
    logger.info("Validation solution file")
    fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df_true = pd.read_csv(fn_true)

    # # Remove prefix "PAN_" from ImageId column
    # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test)
    df_test_image_ids = df_test.ImageId.unique()

    fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
        for idx, r in df_true.iterrows():
            f.write("{},{},\"{}\",{:.6f}\n".format(
                r.ImageId,
                r.BuildingId,
                r.PolygonWKT_Pix,
                1.0))


def _internal_validate_fscore(area_id,
                              epoch=3,
                              predict=True,
                              min_th=MIN_POLYGON_AREA,
                              enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)

    # ------------------------
    # Prediction phase
    logger.info("Prediction phase")
    if predict:
        _internal_validate_predict(
            area_id,
            epoch=epoch,
            enable_tqdm=enable_tqdm)

    # ------------------------
    # Postprocessing phase
    logger.info("Postprocessing phase")
    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')
    fn = FMT_VALTESTPRED_PATH.format(prefix)
    with tb.open_file(fn, 'r') as f:
        y_pred = np.array(f.get_node('/pred'))

    fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        test_list = df_test.index.tolist()
        iterator = enumerate(test_list)

        for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
            pred_values = np.zeros((650, 650))
            pred_count = np.zeros((650, 650))
            for slice_pos in range(9):
                slice_idx = idx * 9 + slice_pos

                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = STRIDE_SZ * pos_i
                y0 = STRIDE_SZ * pos_j
                pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
                    y_pred[slice_idx][0]
                )
                pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1
            pred_values = pred_values / pred_count

            df_poly = mask_to_poly(pred_values, min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = _remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))

    # ------------------------
    # Validation solution file
    logger.info("Validation solution file")
    fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df_true = pd.read_csv(fn_true)

    # # Remove prefix "PAN_" from ImageId column
    # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test)
    df_test_image_ids = df_test.ImageId.unique()

    fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
        for idx, r in df_true.iterrows():
            f.write("{},{},\"{}\",{:.6f}\n".format(
                r.ImageId,
                r.BuildingId,
                r.PolygonWKT_Pix,
                1.0))


def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
    mask = (mask > 0.5).astype(np.uint8)
    shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
    poly_list = []
    mp = shapely.ops.cascaded_union(
        shapely.geometry.MultiPolygon([
            shapely.geometry.shape(shape)
            for shape, value in shapes
        ]))

    if isinstance(mp, shapely.geometry.Polygon):
        df = pd.DataFrame({
            'area_size': [mp.area],
            'poly': [mp],
        })
    else:
        df = pd.DataFrame({
            'area_size': [p.area for p in mp],
            'poly': [p for p in mp],
        })

    df = df[df.area_size > min_polygon_area_th].sort_values(
        by='area_size', ascending=False)
    df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
        x, rounding_precision=0))
    df.loc[:, 'bid'] = list(range(1, len(df) + 1))
    df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
    return df


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def generate_test_batch(area_id,
                        batch_size=64,
                        immean=None,
                        enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    df_test = pd.read_csv(FMT_TEST_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_TEST_MUL_STORE.format(prefix)
    fn_osm = FMT_TEST_OSM_STORE.format(prefix)

    slice_id_list = []
    for idx, row in df_test.iterrows():
        for slice_pos in range(9):
            slice_id = row.ImageId + '_' + str(slice_pos)
            slice_id_list.append(slice_id)

    if enable_tqdm:
        pbar = tqdm.tqdm(total=len(slice_id_list))

    while 1:
        total_sz = len(slice_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im,\
                tb.open_file(fn_osm, 'r') as f_osm:
            for i_batch in range(n_batch):
                target_slice_ids = slice_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_slice_ids) == 0:
                    continue

                X_test = []
                y_test = []
                for slice_id in target_slice_ids:
                    im = np.array(f_im.get_node('/' + slice_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    im2 = np.array(f_osm.get_node('/' + slice_id))
                    im2 = np.swapaxes(im2, 0, 2)
                    im2 = np.swapaxes(im2, 1, 2)
                    im = np.vstack([im, im2])

                    X_test.append(im)
                    mask = np.zeros((INPUT_SIZE, INPUT_SIZE)).astype(np.uint8)
                    y_test.append(mask)
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                y_test = y_test.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_test = X_test - immean

                if enable_tqdm:
                    pbar.update(y_test.shape[0])

                yield (X_test, y_test)

        if enable_tqdm:
            pbar.close()


def generate_valtest_batch(area_id,
                           batch_size=8,
                           immean=None,
                           enable_tqdm=False):
    prefix = area_id_to_prefix(area_id)
    df_train = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
    fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)
    fn_osm = FMT_VALTEST_OSM_STORE.format(prefix)

    slice_id_list = []
    for idx, row in df_train.iterrows():
        for slice_pos in range(9):
            slice_id = row.ImageId + '_' + str(slice_pos)
            slice_id_list.append(slice_id)

    if enable_tqdm:
        pbar = tqdm.tqdm(total=len(slice_id_list))

    while 1:
        total_sz = len(slice_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im,\
                tb.open_file(fn_osm, 'r') as f_osm,\
                tb.open_file(fn_mask, 'r') as f_mask:
            for i_batch in range(n_batch):
                target_slice_ids = slice_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_slice_ids) == 0:
                    continue

                X_train = []
                y_train = []
                for slice_id in target_slice_ids:
                    im = np.array(f_im.get_node('/' + slice_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    im2 = np.array(f_osm.get_node('/' + slice_id))
                    im2 = np.swapaxes(im2, 0, 2)
                    im2 = np.swapaxes(im2, 1, 2)
                    im = np.vstack([im, im2])

                    X_train.append(im)
                    mask = np.array(f_mask.get_node('/' + slice_id))
                    mask = (mask > 0).astype(np.uint8)
                    y_train.append(mask)
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                y_train = y_train.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_train = X_train - immean

                if enable_tqdm:
                    pbar.update(y_train.shape[0])

                yield (X_train, y_train)

        if enable_tqdm:
            pbar.close()


def generate_valtrain_batch(area_id, batch_size=8, immean=None):
    prefix = area_id_to_prefix(area_id)
    df_train = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
    fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)
    fn_osm = FMT_VALTRAIN_OSM_STORE.format(prefix)

    slice_id_list = []
    for idx, row in df_train.iterrows():
        for slice_pos in range(9):
            slice_id = row.ImageId + '_' + str(slice_pos)
            slice_id_list.append(slice_id)
    np.random.shuffle(slice_id_list)

    while 1:
        total_sz = len(slice_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im,\
                tb.open_file(fn_osm, 'r') as f_osm,\
                tb.open_file(fn_mask, 'r') as f_mask:
            for i_batch in range(n_batch):
                target_slice_ids = slice_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_slice_ids) == 0:
                    continue

                X_train = []
                y_train = []
                for slice_id in target_slice_ids:
                    im = np.array(f_im.get_node('/' + slice_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    im2 = np.array(f_osm.get_node('/' + slice_id))
                    im2 = np.swapaxes(im2, 0, 2)
                    im2 = np.swapaxes(im2, 1, 2)
                    im = np.vstack([im, im2])

                    X_train.append(im)
                    mask = np.array(f_mask.get_node('/' + slice_id))
                    mask = (mask > 0).astype(np.uint8)
                    y_train.append(mask)
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                y_train = y_train.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_train = X_train - immean

                yield (X_train, y_train)


def get_unet(input_layers=15):
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(mode='concat', concat_axis=1)
    inputs = Input((input_layers, 256, 256))
    conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
    conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
    conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
    conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
    conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
    conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    optimizer = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model


def get_mul_mean_image(area_id):
    prefix = area_id_to_prefix(area_id)

    with tb.open_file(FMT_MULMEAN.format(prefix), 'r') as f:
        im_mean = np.array(f.get_node('/mulmean'))
    return im_mean


def __load_band_cut_th(band_fn, bandsz=3):
    df = pd.read_csv(band_fn, index_col='area_id')
    all_band_cut_th = {area_id: {} for area_id in range(2, 6)}
    for area_id, row in df.iterrows():
        for chan_i in range(bandsz):
            all_band_cut_th[area_id][chan_i] = dict(
                min=row['chan{}_min'.format(chan_i)],
                max=row['chan{}_max'.format(chan_i)],
            )
    return all_band_cut_th


def get_mask_im(df, image_id):
    im_mask = np.zeros((650, 650))
    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = [round(float(pp[0])) for pp in coords]
            y = [round(float(pp[1])) for pp in coords]
            yy, xx = skimage.draw.polygon(y, x, (650, 650))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 0
    im_mask = (im_mask > 0.5).astype(np.uint8)
    return im_mask


def get_slice_mask_im(df, image_id):
    im_mask = np.zeros((650, 650))
    for idx, row in df[df.ImageId == image_id].iterrows():
        shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
        if shape_obj.exterior is not None:
            coords = list(shape_obj.exterior.coords)
            x = [round(float(pp[0])) for pp in coords]
            y = [round(float(pp[1])) for pp in coords]
            yy, xx = skimage.draw.polygon(y, x, (650, 650))
            im_mask[yy, xx] = 1

            interiors = shape_obj.interiors
            for interior in interiors:
                coords = list(interior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 0
    im_mask = (im_mask > 0.5).astype(np.uint8)

    for slice_pos in range(9):
        pos_j = int(math.floor(slice_pos / 3.0))
        pos_i = int(slice_pos % 3)
        x0 = STRIDE_SZ * pos_i
        y0 = STRIDE_SZ * pos_j
        im_mask_part = im_mask[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
        assert im_mask_part.shape == (256, 256)
        yield slice_pos, im_mask_part


def get_test_image_path_from_imageid(image_id, datapath, mul=False):
    if mul:
        return FMT_TEST_MSPEC_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)
    else:
        return FMT_TEST_RGB_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)


def get_train_image_path_from_imageid(image_id, datapath, mul=False):
    if mul:
        return FMT_TRAIN_MSPEC_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)
    else:
        return FMT_TRAIN_RGB_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)


def image_id_to_prefix(image_id):
    prefix = image_id.split('img')[0][:-1]
    return prefix


def load_train_summary_data(area_id):
    prefix = area_id_to_prefix(area_id)
    fn = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df = pd.read_csv(fn)
    # df.loc[:, 'ImageId'] = df.ImageId.str[4:]
    return df


def split_val_train_test(area_id):
    prefix = area_id_to_prefix(area_id)

    df = load_train_summary_data(area_id)
    df_agg = df.groupby('ImageId').agg('first')
    image_id_list = df_agg.index.tolist()
    np.random.shuffle(image_id_list)
    sz_valtrain = int(len(image_id_list) * 0.7)
    sz_valtest = len(image_id_list) - sz_valtrain

    # Parent directory
    parent_dir = Path(FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)).parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True)

    pd.DataFrame({'ImageId': image_id_list[:sz_valtrain]}).to_csv(
        FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
        index=False)
    pd.DataFrame({'ImageId': image_id_list[sz_valtrain:]}).to_csv(
        FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
        index=False)


# ---------------------------------------------------------


def calc_multiband_cut_threshold(path_list):
    band_values = {k: [] for k in range(3)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(3)}
    for path in path_list:
        with rasterio.open(path, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(3):
                band_values[i_chan].append(values[i_chan].ravel())
    for i_chan in range(3):
        band_values[i_chan] = np.concatenate(band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(band_values[i_chan], 2)
    return band_cut_th


def tif_to_latlon(path):
    ds = gdal.Open(path)
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width*gt[4] + height*gt[5]
    maxx = gt[0] + width*gt[1] + height*gt[2]
    maxy = gt[3]
    return Bunch(
        minx=minx,
        maxx=maxx,
        miny=miny,
        maxy=maxy,
        xcenter=(minx+maxx)/2.0,
        ycenter=(miny+maxy)/2.0)


def location_summary(area_id, datapath):
    area_prefix = area_id_to_prefix(area_id)
    rows = []

    glob_path = str(
        Path(datapath) /
        Path("PAN/PAN_{prefix:s}_img*.tif")
    ).format(prefix=area_prefix)

    for path in sorted(glob.glob(glob_path)):
        image_id = path.split('/')[-1][:-4]
        pos = tif_to_latlon(path)
        rows.append(dict(ImageId=image_id, path=path, pos=pos))

    df_location = pd.DataFrame(rows)
    df_location.loc[:, 'xcenter'] = df_location.pos.apply(lambda x: x.xcenter)
    df_location.loc[:, 'ycenter'] = df_location.pos.apply(lambda x: x.ycenter)
    return df_location


def location_summary_test(area_id, datapath):
    area_prefix = area_id_to_prefix(area_id)
    rows = []
    glob_path = str(
        Path(datapath) /
        Path("PAN/PAN_{prefix:s}_img*.tif")
    ).format(prefix=area_prefix)

    for path in sorted(glob.glob(glob_path)):
        image_id = path.split('/')[-1][:-4]
        pos = tif_to_latlon(path)
        rows.append(dict(ImageId=image_id, path=path, pos=pos))

    df_location = pd.DataFrame(rows)
    df_location.loc[:, 'xcenter'] = df_location.pos.apply(lambda x: x.xcenter)
    df_location.loc[:, 'ycenter'] = df_location.pos.apply(lambda x: x.ycenter)
    return df_location


def get_mapzen_osm_name(area_id):
    area_id_to_mapzen_name = {
        2: 'las-vegas_nevada_osm',
        3: 'paris_france_osm',
        4: 'shanghai_china_osm',
        5: 'ex_s2cCo6gpCXAvihWVygCAfSjNVksnQ_osm',
    }
    mapzen_name = area_id_to_mapzen_name[area_id]
    return mapzen_name


def extract_buildings_geoms(area_id):
    mapzen_name = get_mapzen_osm_name(area_id)
    fn_osm = FMT_SERIALIZED_OSMDATA.format(mapzen_name)
    with open(fn_osm, 'rb') as f:
        osm = pickle.load(f)

    geoms = [
        geom
        for geom, type_name, properties in osm['buildings']
        if type_name == 'area'
    ]
    return geoms


def extract_waterarea_geoms(area_id):
    mapzen_name = get_mapzen_osm_name(area_id)
    fn_osm = FMT_SERIALIZED_OSMDATA.format(mapzen_name)
    with open(fn_osm, 'rb') as f:
        osm = pickle.load(f)

    geoms = [
        geom
        for geom, type_name, properties in osm['waterareas']
        if type_name == 'area'
    ]
    return geoms


def extract_landusages_industrial_geoms(area_id):
    mapzen_name = get_mapzen_osm_name(area_id)
    fn_osm = FMT_SERIALIZED_OSMDATA.format(mapzen_name)
    with open(fn_osm, 'rb') as f:
        osm = pickle.load(f)

    geoms = [
        geom
        for geom, type_name, properties in osm['landusages']
        if type_name == 'area' and properties['type'] == 'industrial'
    ]
    return geoms


def extract_landusages_farm_and_forest_geoms(area_id):
    mapzen_name = get_mapzen_osm_name(area_id)
    fn_osm = FMT_SERIALIZED_OSMDATA.format(mapzen_name)
    with open(fn_osm, 'rb') as f:
        osm = pickle.load(f)

    geoms = [
        geom
        for geom, type_name, properties in osm['landusages']
        if type_name == 'area' and properties['type'] in [
            'forest',
            'farmyard',
        ]
    ]
    return geoms


def extract_landusages_residential_geoms(area_id):
    mapzen_name = get_mapzen_osm_name(area_id)
    fn_osm = FMT_SERIALIZED_OSMDATA.format(mapzen_name)
    with open(fn_osm, 'rb') as f:
        osm = pickle.load(f)

    geoms = [
        geom
        for geom, type_name, properties in osm['landusages']
        if type_name == 'area' and properties['type'] == 'residential'
    ]
    return geoms


def extract_roads_geoms(area_id):
    mapzen_name = get_mapzen_osm_name(area_id)
    fn_osm = FMT_SERIALIZED_OSMDATA.format(mapzen_name)
    with open(fn_osm, 'rb') as f:
        osm = pickle.load(f)

    geoms = [
        geom
        for geom, type_name, properties in osm['roads']
        if type_name == 'line' and properties['type'] != 'subway'
    ]
    return geoms


def extract_osmlayers(area_id):
    if area_id == 2:
        return [
            extract_buildings_geoms(area_id),
            extract_landusages_industrial_geoms(area_id),
            extract_landusages_residential_geoms(area_id),
            extract_roads_geoms(area_id),
        ]
    elif area_id == 3:
        return [
            extract_buildings_geoms(area_id),
            extract_landusages_farm_and_forest_geoms(area_id),
            extract_landusages_industrial_geoms(area_id),
            extract_landusages_residential_geoms(area_id),
            extract_roads_geoms(area_id),
        ]
    elif area_id == 4:
        return [
            extract_waterarea_geoms(area_id),
            extract_landusages_industrial_geoms(area_id),
            extract_landusages_residential_geoms(area_id),
            extract_roads_geoms(area_id),
        ]
    elif area_id == 5:
        return [
            extract_waterarea_geoms(area_id),
            extract_landusages_industrial_geoms(area_id),
            extract_landusages_residential_geoms(area_id),
            extract_roads_geoms(area_id),
        ]
    else:
        raise RuntimeError("area_id must be in range(2, 6): {}".foramt(
            area_id))


def prep_osmlayer_test(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    logger.info("prep_osmlayer_test for {}".format(prefix))

    fn_list = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    fn_store = FMT_TEST_OSM_STORE.format(prefix)

    layers = extract_osmlayers(area_id)

    df = pd.read_csv(fn_list, index_col='ImageId')
    logger.info("Prep osm container: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        df_sz = len(df)
        for image_id in tqdm.tqdm(df.index, total=df_sz):
            # fn_tif = test_image_id_to_path(image_id)
            fn_tif = get_test_image_path_from_imageid(
                image_id, datapath, mul=False)
            with rasterio.open(fn_tif, 'r') as fr:
                values = fr.read(1)
                masks = []  # rasterize masks
                for layer_geoms in layers:
                    mask = rasterio.features.rasterize(
                        layer_geoms,
                        out_shape=values.shape,
                        transform=rasterio.guard_transform(
                            fr.transform))
                    masks.append(mask)
                masks = np.array(masks)
                masks = np.swapaxes(masks, 0, 2)
                masks = np.swapaxes(masks, 0, 1)
            assert masks.shape == (650, 650, len(layers))

            # slice of masks
            for slice_pos in range(9):
                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = STRIDE_SZ * pos_i
                y0 = STRIDE_SZ * pos_j
                im = masks[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
                assert im.shape == (256, 256, len(layers))

                slice_id = image_id + "_{}".format(slice_pos)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root,
                                     slice_id,
                                     atom,
                                     im.shape,
                                     filters=filters)
                ds[:] = im


def prep_osmlayer_train(area_id, datapath, is_valtrain=False):
    prefix = area_id_to_prefix(area_id)
    logger.info("prep_osmlayer_train for {}".format(prefix))

    if is_valtrain:
        fn_list = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTRAIN_OSM_STORE.format(prefix)
    else:
        fn_list = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTEST_OSM_STORE.format(prefix)

    layers = extract_osmlayers(area_id)

    df = pd.read_csv(fn_list, index_col='ImageId')
    logger.info("Prep osm container: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        df_sz = len(df)
        for image_id in tqdm.tqdm(df.index, total=df_sz):
            # fn_tif = train_image_id_to_path(image_id)
            fn_tif = get_train_image_path_from_imageid(
                image_id, datapath, mul=False)
            with rasterio.open(fn_tif, 'r') as fr:
                values = fr.read(1)
                masks = []  # rasterize masks
                for layer_geoms in layers:
                    mask = rasterio.features.rasterize(
                        layer_geoms,
                        out_shape=values.shape,
                        transform=rasterio.guard_transform(
                            fr.transform))
                    masks.append(mask)
                masks = np.array(masks)
                masks = np.swapaxes(masks, 0, 2)
                masks = np.swapaxes(masks, 0, 1)
            assert masks.shape == (650, 650, len(layers))

            # slice of masks
            for slice_pos in range(9):
                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = STRIDE_SZ * pos_i
                y0 = STRIDE_SZ * pos_j
                im = masks[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
                assert im.shape == (256, 256, len(layers))

                slice_id = image_id + "_{}".format(slice_pos)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root,
                                     slice_id,
                                     atom,
                                     im.shape,
                                     filters=filters)
                ds[:] = im


def preproc_osm(area_id, datapath, is_train=True):
    logger.info("Loading raster...")
    osmprefix = area_id_to_osmprefix(area_id)

    # df = pd.concat([
    #     location_summary(area_id),
    #     location_summary_test(area_id),
    # ])
    if is_train:
        df = location_summary(area_id, datapath)
    else:
        df = location_summary_test(area_id, datapath)

    map_bound = Bunch(
        left=df.sort_values(by='xcenter').iloc[-1]['pos']['maxx'],
        right=df.sort_values(by='xcenter').iloc[0]['pos']['minx'],
        top=df.sort_values(by='ycenter').iloc[-1]['pos']['maxy'],
        bottom=df.sort_values(by='ycenter').iloc[0]['pos']['miny'],
    )
    geom_layers = {}

    fn_osm = FMT_SERIALIZED_OSMDATA.format(osmprefix)
    if not Path(fn_osm).exists():
        for layer_name in LAYER_NAMES:
            fn_shp = FMT_OSMSHAPEFILE.format(
                name=osmprefix,
                layer=layer_name)

            if not Path(fn_shp).exists():
                raise RuntimeError("shp not found: {}".format(fn_shp))

            geom_bounds = shapely.geometry.Polygon([
                (map_bound.left, map_bound.top),
                (map_bound.right, map_bound.top),
                (map_bound.right, map_bound.bottom),
                (map_bound.left, map_bound.bottom),
            ])
            with fiona.open(fn_shp, 'r') as vector:
                print("{}: {}".format(layer_name, len(vector)))
                geoms = []
                for feat in tqdm.tqdm(vector, total=len(vector)):
                    try:
                        geom = shapely.geometry.shape(feat['geometry'])
                        isec_area = geom.intersection(geom_bounds).area
                        if isec_area > 0:
                            geoms.append([
                                geom, 'area', feat['properties'],
                            ])
                        elif geom.intersects(geom_bounds):
                            geoms.append([
                                geom, 'line', feat['properties'],
                            ])
                    except:
                        pass

                print("{}: {} -> {}".format(
                    layer_name,
                    len(vector),
                    len(geoms)))
                geom_layers[layer_name] = geoms

        with open(fn_osm, 'wb') as f:
            pickle.dump(geom_layers, f)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--testonly/--no-testonly', default=True)
def testmerge(testonly):
    # file check: test
    for area_id in range(2, 6):
        prefix = area_id_to_prefix(area_id)
        fn_out = FMT_TESTPOLY_PATH.format(prefix)
        if not Path(fn_out).exists():
            logger.info("Required file not found: {}".format(fn_out))
            sys.exit(1)

    if not testonly:
        # file check: valtest
        for area_id in range(2, 6):
            prefix = area_id_to_prefix(area_id)
            fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
            if not Path(fn_out).exists():
                logger.info("Required file not found: {}".format(fn_out))
                sys.exit(1)

    # merge files: test poly
    rows = []
    for area_id in range(2, 6):
        prefix = area_id_to_prefix(area_id)
        fn_out = FMT_TESTPOLY_PATH.format(prefix)
        with open(fn_out, 'r') as f:
            line = f.readline()
            if area_id == 2:
                rows.append(line)
            for line in f:
                line = _remove_interiors(line)
                rows.append(line)
    with open(FN_SOLUTION_CSV, 'w') as f:
        for line in rows:
            f.write(line)

    if not testonly:
        # merge files: valtest poly
        rows = []
        for area_id in range(2, 6):
            prefix = area_id_to_prefix(area_id)
            fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
            with open(fn_out, 'r') as f:
                line = f.readline()
                if area_id == 2:
                    rows.append(line)
                for line in f:
                    line = _remove_interiors(line)
                    rows.append(line)
        fn_out = FMT_VALTESTPOLY_OVALL_PATH
        with open(fn_out, 'w') as f:
            for line in rows:
                f.write(line)

        # merge files: valtest truth
        rows = []
        for area_id in range(2, 6):
            prefix = area_id_to_prefix(area_id)
            fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
            with open(fn_out, 'r') as f:
                line = f.readline()
                if area_id == 2:
                    rows.append(line)
                for line in f:
                    rows.append(line)
        fn_out = FMT_VALTESTTRUTH_OVALL_PATH
        with open(fn_out, 'w') as f:
            for line in rows:
                f.write(line)


@cli.command()
@click.argument('area_id', type=int)
def testproc(area_id):
    prefix = area_id_to_prefix(area_id)
    logger.info(">>>> Test proc for {}".format(prefix))

    _internal_test(area_id)
    logger.info(">>>> Test proc for {} ... done".format(prefix))


@cli.command()
@click.argument('area_id', type=int)
@click.option('--epoch', type=int, default=0)
@click.option('--th', type=int, default=MIN_POLYGON_AREA)
@click.option('--predict/--no-predict', default=False)
def validate_city_fscore(area_id, epoch, th, predict):
    _internal_validate_fscore(
        area_id,
        epoch=epoch,
        enable_tqdm=True,
        min_th=th,
        predict=predict)
    evaluate_record = _calc_fscore_per_aoi(area_id)
    evaluate_record['epoch'] = epoch
    evaluate_record['min_area_th'] = th
    evaluate_record['area_id'] = area_id
    logger.info("\n" + json.dumps(evaluate_record, indent=4))


@cli.command()
@click.argument('datapath', type=str)
def evalfscore(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("Evaluate fscore on validation set: {}".format(prefix))

    # for each epoch
    # if not Path(FMT_VALMODEL_EVALHIST.format(prefix)).exists():
    if True:
        df_hist = pd.read_csv(FMT_VALMODEL_HIST.format(prefix))
        df_hist.loc[:, 'epoch'] = list(range(1, len(df_hist) + 1))

        rows = []
        for zero_base_epoch in range(0, len(df_hist)):
            logger.info(">>> Epoch: {}".format(zero_base_epoch))

            _internal_validate_fscore_wo_pred_file(
                area_id,
                epoch=zero_base_epoch,
                enable_tqdm=True,
                min_th=MIN_POLYGON_AREA)
            evaluate_record = _calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = zero_base_epoch
            evaluate_record['min_area_th'] = MIN_POLYGON_AREA
            evaluate_record['area_id'] = area_id
            logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(
            FMT_VALMODEL_EVALHIST.format(prefix),
            index=False)

    # find best min-poly-threshold
    df_evalhist = pd.read_csv(FMT_VALMODEL_EVALHIST.format(prefix))
    best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
    best_epoch = int(best_row.zero_base_epoch)
    best_fscore = best_row.fscore

    # optimize min area th
    rows = []
    for th in [30, 60, 90, 120, 150, 180, 210, 240]:
        logger.info(">>> TH: {}".format(th))
        predict_flag = False
        if th == 30:
            predict_flag = True

        _internal_validate_fscore(
            area_id,
            epoch=best_epoch,
            enable_tqdm=True,
            min_th=th,
            predict=predict_flag)
        evaluate_record = _calc_fscore_per_aoi(area_id)
        evaluate_record['zero_base_epoch'] = best_epoch
        evaluate_record['min_area_th'] = th
        evaluate_record['area_id'] = area_id
        logger.info("\n" + json.dumps(evaluate_record, indent=4))
        rows.append(evaluate_record)

    pd.DataFrame(rows).to_csv(
        FMT_VALMODEL_EVALTHHIST.format(prefix),
        index=False)

    logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def validate(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info(">> validate sub-command: {}".format(prefix))

    dict_n_osm_layers = {
        2: 4,
        3: 5,
        4: 4,
        5: 4,
    }
    osm_layers = dict_n_osm_layers[area_id]
    n_input_layers = 8 + osm_layers

    prefix = area_id_to_prefix(area_id)
    logger.info("Validate step for {}".format(prefix))
    X_mean = get_mul_mean_image(area_id)
    X_osm_mean = np.zeros((osm_layers, INPUT_SIZE, INPUT_SIZE))
    X_mean = np.vstack([X_mean, X_osm_mean])

    if not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True)

    logger.info("load valtrain")

    logger.info("Instantiate U-Net model")
    model = get_unet(input_layers=n_input_layers)
    model_checkpoint = ModelCheckpoint(
        FMT_VALMODEL_PATH.format(prefix + "_{epoch:02d}"),
        monitor='val_jaccard_coef_int',
        save_best_only=False)
    model_earlystop = EarlyStopping(
        monitor='val_jaccard_coef_int',
        patience=10,
        verbose=0,
        mode='max')
    model_history = History()

    df_train = pd.read_csv(
        FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
    df_test = pd.read_csv(
        FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix))

    logger.info("Fit")
    model.fit_generator(
        generate_valtrain_batch(area_id, batch_size=2, immean=X_mean),
        samples_per_epoch=len(df_train) * 9,
        nb_epoch=35,
        verbose=1,
        validation_data=generate_valtest_batch(
            area_id,
            batch_size=2,
            immean=X_mean,
        ),
        nb_val_samples=len(df_test) * 9,
        callbacks=[model_checkpoint, model_earlystop, model_history])

    model.save_weights(FMT_VALMODEL_LAST_PATH.format(prefix))

    # Save evaluation history
    pd.DataFrame(model_history.history).to_csv(
        FMT_VALMODEL_HIST.format(prefix), index=False)
    logger.info(">> validate sub-command: {} ... Done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def preproc_train(datapath):
    """ train.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    osmprefix = area_id_to_osmprefix(area_id)

    # Mkdir
    if not Path(FMT_VALTRAIN_OSM_STORE.format(prefix)).parent.exists():
        Path(FMT_VALTRAIN_OSM_STORE.format(prefix)).parent.mkdir(
                parents=True)

    # OSM serialized subset
    fn_osm = FMT_SERIALIZED_OSMDATA.format(osmprefix)
    if Path(fn_osm).exists():
        logger.info("Serialize OSM subset ... skip")
    else:
        logger.info("Serialize OSM subset")
        preproc_osm(area_id, datapath, is_train=True)

    # OSM layers (valtrain)
    if Path(FMT_VALTRAIN_OSM_STORE.format(prefix)).exists():
        logger.info("Generate OSM_STORE (valtrain) ... skip")
    else:
        logger.info("Generate OSM_STORE (valtrain)")
        prep_osmlayer_train(area_id, datapath, is_valtrain=True)

    # OSM layers (valtest)
    if Path(FMT_VALTEST_OSM_STORE.format(prefix)).exists():
        logger.info("Generate OSM_STORE (valtest) ... skip")
    else:
        logger.info("Generate OSM_STORE (valtest)")
        prep_osmlayer_train(area_id, datapath, is_valtrain=False)


@cli.command()
@click.argument('datapath', type=str)
def preproc_test(datapath):
    """ test.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    osmprefix = area_id_to_osmprefix(area_id)

    # Mkdir
    if not Path(FMT_TEST_OSM_STORE.format(prefix)).parent.exists():
        Path(FMT_TEST_OSM_STORE.format(prefix)).parent.mkdir(
                parents=True)

    # OSM serialized subset
    fn_osm = FMT_SERIALIZED_OSMDATA.format(osmprefix)
    if Path(fn_osm).exists():
        logger.info("Serialize OSM subset ... skip")
    else:
        logger.info("Serialize OSM subset")
        preproc_osm(area_id, datapath, is_train=False)

    # OSM layers (test)
    if Path(FMT_TEST_OSM_STORE.format(prefix)).exists():
        logger.info("Generate OSM_STORE (test) ... skip")
    else:
        logger.info("Generate OSM_STORE (test)")
        prep_osmlayer_test(area_id, datapath)


if __name__ == '__main__':
    cli()
