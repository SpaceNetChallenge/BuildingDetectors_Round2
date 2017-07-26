# -*- coding: utf-8 -*-
"""
v17.csv (final submission) ... averaging model of v9s, v13 and v16
"""
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
import subprocess
import importlib
import math
from pathlib import Path
import json
import re
import warnings

import tqdm
import click
import tables as tb
import pandas as pd
import numpy as np
import shapely.ops
import shapely.geometry
import skimage.transform
import rasterio.features


MODEL_NAME = 'v17'
ORIGINAL_SIZE = 650
INPUT_SIZE = 256
STRIDE_SZ = 197

LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
BASE_DIR = "/data/train"

# Parameters
MIN_POLYGON_AREA = 30

# Input files (required for validation)
FMT_TRAIN_SUMMARY_PATH = str(
    Path(BASE_DIR) /
    Path("{prefix:s}_Train/") /
    Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))

# ---------------------------------------------------------
# Image list, Image container and mask container
V5_IMAGE_DIR = "/data/working/images/{}".format('v5')
FMT_VALTEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_TEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"

# Model files
MODEL_DIR = "/data/working/models/{}".format(MODEL_NAME)
FMT_VALTESTPRED_PATH = MODEL_DIR + "/{}_eval_pred.h5"
FMT_VALTESTPOLY_PATH = MODEL_DIR + "/{}_eval_poly.csv"
FMT_VALTESTTRUTH_PATH = MODEL_DIR + "/{}_eval_poly_truth.csv"
FMT_VALTESTPOLY_OVALL_PATH = MODEL_DIR + "/eval_poly.csv"
FMT_VALTESTTRUTH_OVALL_PATH = MODEL_DIR + "/eval_poly_truth.csv"
FMT_VALMODEL_EVALTHHIST = MODEL_DIR + "/{}_val_evalhist_th.csv"

# ---------------------------------------------------------
# Prediction & polygon result
FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"
FN_SOLUTION_CSV = "/data/{}.csv".format(MODEL_NAME)

# Logger
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter(LOGFORMAT))
fh_handler = FileHandler(".{}.log".format(MODEL_NAME))
fh_handler.setFormatter(Formatter(LOGFORMAT))
logger = getLogger('spacenet2')
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)
    logger.addHandler(fh_handler)


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


def get_model_parameter(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_hist = FMT_VALMODEL_EVALTHHIST.format(prefix)
    best_row = pd.read_csv(fn_hist).sort_values(
        by='fscore',
        ascending=False,
    ).iloc[0]

    param = dict(
        min_poly_area=int(best_row['min_area_th']),
    )
    return param


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


def area_id_to_prefix(area_id):
    """
    area_id から prefix を返す
    """
    area_dict = {
        2: 'AOI_2_Vegas',
        3: 'AOI_3_Paris',
        4: 'AOI_4_Shanghai',
        5: 'AOI_5_Khartoum',
    }
    return area_dict[area_id]


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

    # Expected lines:
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


def extract_y_pred(mod, area_id):
    prefix = area_id_to_prefix(area_id)
    fn = mod.FMT_VALTESTPRED_PATH.format(prefix)
    with tb.open_file(fn, 'r') as f:
        y_pred = np.array(f.get_node('/pred'))

    return y_pred


def _internal_test_predict_best_param(area_id,
                                      rescale_pred_list=[],
                                      slice_pred_list=[]):
    prefix = area_id_to_prefix(area_id)

    # Load test imagelist
    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    pred_values_array = np.zeros((len(df_test), 650, 650))
    for idx, image_id in enumerate(df_test.index.tolist()):
        pred_values = np.zeros((650, 650))
        pred_count = np.zeros((650, 650))
        for slice_pos in range(9):
            slice_idx = idx * 9 + slice_pos

            pos_j = int(math.floor(slice_pos / 3.0))
            pos_i = int(slice_pos % 3)
            x0 = STRIDE_SZ * pos_i
            y0 = STRIDE_SZ * pos_j

            for slice_pred in slice_pred_list:
                pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
                    slice_pred[slice_idx][0])
                pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1

        for rescale_pred in rescale_pred_list:
            y_pred_idx = skimage.transform.resize(
                rescale_pred[idx][0], (650, 650))
            pred_values += y_pred_idx
        pred_count += 1

        # Normalize
        pred_values = pred_values / pred_count
        pred_values_array[idx, :, :] = pred_values

    return pred_values_array


def _internal_validate_predict_best_param(area_id,
                                          save_pred=True,
                                          enable_tqdm=False,
                                          rescale_pred_list=[],
                                          slice_pred_list=[]):
    prefix = area_id_to_prefix(area_id)

    # Load valtest imagelist
    fn_valtest = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_valtest = pd.read_csv(fn_valtest, index_col='ImageId')

    pred_values_array = np.zeros((len(df_valtest), 650, 650))
    for idx, image_id in enumerate(df_valtest.index.tolist()):
        pred_values = np.zeros((650, 650))
        pred_count = np.zeros((650, 650))
        for slice_pos in range(9):
            slice_idx = idx * 9 + slice_pos

            pos_j = int(math.floor(slice_pos / 3.0))
            pos_i = int(slice_pos % 3)
            x0 = STRIDE_SZ * pos_i
            y0 = STRIDE_SZ * pos_j

            for slice_pred in slice_pred_list:
                pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
                    slice_pred[slice_idx][0])
                pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1

        for rescale_pred in rescale_pred_list:
            y_pred_idx = skimage.transform.resize(
                rescale_pred[idx][0], (650, 650))
            pred_values += y_pred_idx
        pred_count += 1

        # Normalize
        pred_values = pred_values / pred_count
        pred_values_array[idx, :, :] = pred_values

    return pred_values_array


def _internal_pred_to_poly_file_test(area_id,
                                     y_pred,
                                     min_th=MIN_POLYGON_AREA):
    """
    Write out test poly
    """
    prefix = area_id_to_prefix(area_id)

    # Load test imagelist
    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    # Make parent directory
    fn_out = FMT_TESTPOLY_PATH.format(prefix)
    if not Path(fn_out).parent.exists():
        Path(fn_out).parent.mkdir(parents=True)

    # Ensemble individual models and write out output files
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        for idx, image_id in enumerate(df_test.index.tolist()):
            df_poly = mask_to_poly(y_pred[idx], min_polygon_area_th=min_th)
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


def _internal_pred_to_poly_file(area_id,
                                y_pred,
                                min_th=MIN_POLYGON_AREA):
    """
    Write out valtest poly and truepoly
    """
    prefix = area_id_to_prefix(area_id)

    # Load valtest imagelist
    fn_valtest = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_valtest = pd.read_csv(fn_valtest, index_col='ImageId')

    # Make parent directory
    fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
    if not Path(fn_out).parent.exists():
        Path(fn_out).parent.mkdir(parents=True)

    # Ensemble individual models and write out output files
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        for idx, image_id in enumerate(df_valtest.index.tolist()):
            df_poly = mask_to_poly(y_pred[idx], min_polygon_area_th=min_th)
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

    # Validation solution file
    fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df_true = pd.read_csv(fn_true)

    # # Remove prefix "PAN_"
    # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

    fn_valtest = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_valtest = pd.read_csv(fn_valtest)
    df_valtest_image_ids = df_valtest.ImageId.unique()

    fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
    with open(fn_out, 'w') as f:
        f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
        df_true = df_true[df_true.ImageId.isin(df_valtest_image_ids)]
        for idx, r in df_true.iterrows():
            line = "{},{},\"{}\",{:.6f}\n".format(
                r.ImageId,
                r.BuildingId,
                r.PolygonWKT_Pix,
                1.0)
            f.write(line)


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
@click.argument('datapath', type=str)
def testproc(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info(">>>> Test proc for {}".format(prefix))

    logger.info("import modules")
    v9s = importlib.import_module('v9s')
    v13 = importlib.import_module('v13')
    v16 = importlib.import_module('v16')

    # Predict first
    logger.info("Prediction phase (v9s)")
    y_pred_0 = v9s._internal_test_predict_best_param(
        area_id, save_pred=False)
    logger.info("Prediction phase (v13)")
    y_pred_1 = v13._internal_test_predict_best_param(
        area_id, save_pred=False)
    logger.info("Prediction phase (v16)")
    y_pred_2 = v16._internal_test_predict_best_param(
        area_id, save_pred=False)

    # Ensemble
    logger.info("Averaging")
    y_pred = _internal_test_predict_best_param(
        area_id,
        rescale_pred_list=[y_pred_0],
        slice_pred_list=[y_pred_1, y_pred_2],
    )

    # pred to polygon
    param = get_model_parameter(area_id)
    _internal_pred_to_poly_file_test(
        area_id,
        y_pred,
        min_th=param['min_poly_area'],
    )
    logger.info(">>>> Test proc for {} ... done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def evalfscore(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("Evaluate fscore on validation set: {}".format(prefix))

    logger.info("import modules")
    v9s = importlib.import_module('v9s')
    v13 = importlib.import_module('v13')
    v16 = importlib.import_module('v16')

    # Predict first
    logger.info("Prediction phase")
    y_pred_0 = v9s._internal_validate_predict_best_param(
        area_id, enable_tqdm=True)
    y_pred_1 = v13._internal_validate_predict_best_param(
        area_id, enable_tqdm=True)
    y_pred_2 = v16._internal_validate_predict_best_param(
        area_id, enable_tqdm=True)

    logger.info("Averaging")
    y_pred = _internal_validate_predict_best_param(
        area_id,
        rescale_pred_list=[y_pred_0],
        slice_pred_list=[y_pred_1, y_pred_2],
    )

    # Make parent directory
    fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
    if not Path(fn_out).parent.exists():
        Path(fn_out).parent.mkdir(parents=True)

    # Ensemble individual models and write output files
    rows = []
    for th in [30, 60, 90, 120, 150, 180, 210, 240]:
        logger.info(">>> TH: {}".format(th))

        _internal_pred_to_poly_file(
            area_id,
            y_pred,
            min_th=th)
        evaluate_record = _calc_fscore_per_aoi(area_id)
        evaluate_record['min_area_th'] = th
        evaluate_record['area_id'] = area_id
        logger.info("\n" + json.dumps(evaluate_record, indent=4))
        rows.append(evaluate_record)

    pd.DataFrame(rows).to_csv(
        FMT_VALMODEL_EVALTHHIST.format(prefix),
        index=False)

    logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))


if __name__ == '__main__':
    cli()
