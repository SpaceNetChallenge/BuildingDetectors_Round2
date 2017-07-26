# -*- coding: utf-8 -*-
"""
Image preprocessing module

* 1/1 slice MUL images
* depends on v5_im
* used by v13 and v17

Author: Kohei <i@ho.lc>
"""
from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
import math
import glob
import warnings

import click
import scipy
import tqdm
import tables as tb
import pandas as pd
import numpy as np
import skimage.draw
import rasterio
import shapely.wkt


MODEL_NAME = 'v12'
ORIGINAL_SIZE = 650
INPUT_SIZE = 256
STRIDE_SZ = 197

BASE_DIR = "/data/train"
BASE_TEST_DIR = "/data/test"
WORKING_DIR = "/data/working"
IMAGE_DIR = "/data/working/images/{}".format('v12')
V5_IMAGE_DIR = "/data/working/images/{}".format('v5')

MODEL_DIR = "/data/working/models/{}".format(MODEL_NAME)
FN_SOLUTION_CSV = "/data/output/{}.csv".format(MODEL_NAME)

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

# Preprocessing result
FMT_MUL_BANDCUT_TH_PATH = V5_IMAGE_DIR + "/mul_bandcut{}.csv"

# Image list, Image container and mask container
FMT_VALTRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTRAIN_IM_STORE = IMAGE_DIR + "/valtrain_{}_im.h5"
FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"

FMT_VALTEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_VALTEST_IM_STORE = IMAGE_DIR + "/valtest_{}_im.h5"
FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"

FMT_TRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
FMT_TEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
FMT_TRAIN_IM_STORE = IMAGE_DIR + "/train_{}_im.h5"
FMT_TEST_IM_STORE = IMAGE_DIR + "/test_{}_im.h5"
FMT_TRAIN_MASK_STORE = IMAGE_DIR + "/train_{}_mask.h5"
FMT_TRAIN_MUL_STORE = IMAGE_DIR + "/train_{}_mul.h5"
FMT_TEST_MUL_STORE = IMAGE_DIR + "/test_{}_mul.h5"
FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

# Model files
FMT_VALMODEL_PATH = MODEL_DIR + "/{}_val_weights.h5"
FMT_FULLMODEL_PATH = MODEL_DIR + "/{}_full_weights.h5"
FMT_VALMODEL_HIST = MODEL_DIR + "/{}_val_hist.csv"

# Prediction & polygon result
FMT_TESTPRED_PATH = MODEL_DIR + "/{}_pred.h5"
FMT_VALTESTPRED_PATH = MODEL_DIR + "/{}_eval_pred.h5"
FMT_VALTESTPOLY_PATH = MODEL_DIR + "/{}_eval_poly.csv"
FMT_VALTESTTRUTH_PATH = MODEL_DIR + "/{}_eval_poly_truth.csv"
FMT_VALTESTPOLY_OVALL_PATH = MODEL_DIR + "/eval_poly.csv"
FMT_VALTESTTRUTH_OVALL_PATH = MODEL_DIR + "/eval_poly_truth.csv"
FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"

# Model related files (others)
FMT_VALMODEL_LAST_PATH = MODEL_DIR + "/{}_val_weights_last.h5"
FMT_FULLMODEL_LAST_PATH = MODEL_DIR + "/{}_full_weights_last.h5"

# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('spacenet2')
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)


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


def prefix_to_area_id(prefix):
    area_dict = {
        'AOI_1_Rio': 1,
        'AOI_2_Vegas': 2,
        'AOI_3_Paris': 3,
        'AOI_4_Shanghai': 4,
        'AOI_5_Khartoum': 5,
    }
    return area_dict[area_id]


def area_id_to_prefix(area_id):
    """
    area_id から prefix を返す
    """
    area_dict = {
        1: 'AOI_1_Rio',
        2: 'AOI_2_Vegas',
        3: 'AOI_3_Paris',
        4: 'AOI_4_Shanghai',
        5: 'AOI_5_Khartoum',
    }
    return area_dict[area_id]


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


def __load_mul_bandstats(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_stats = FMT_MUL_BANDCUT_TH_PATH.format(prefix)
    df_stats = pd.read_csv(fn_stats, index_col='area_id')
    r = df_stats.loc[area_id]

    stats_dict = {}
    for chan_i in range(8):
        stats_dict[chan_i] = dict(
            min=r['chan{}_min'.format(chan_i)],
            max=r['chan{}_max'.format(chan_i)])
    return stats_dict


def get_slice_8chan_im(image_id, datapath, bandstats, is_test=False):
    fn = get_train_image_path_from_imageid(
        image_id, datapath, mul=True)
    if is_test:
        fn = get_test_image_path_from_imageid(
            image_id, datapath, mul=True)

    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(8):
            min_val = bandstats[chan_i]['min']
            max_val = bandstats[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
    values = np.swapaxes(values, 0, 2)
    values = np.swapaxes(values, 0, 1)
    assert values.shape == (650, 650, 8)

    for slice_pos in range(9):
        pos_j = int(math.floor(slice_pos / 3.0))
        pos_i = int(slice_pos % 3)
        x0 = STRIDE_SZ * pos_i
        y0 = STRIDE_SZ * pos_j
        im = values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
        assert im.shape == (256, 256, 8)
        yield slice_pos, im


def get_slice_mask_im(df, image_id):
    im_mask = np.zeros((650, 650))

    if len(df[df.ImageId == image_id]) == 0:
        raise RuntimeError("ImageId not found on summaryData: {}".format(
            image_id))

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


def prep_image_mask(area_id, is_valtrain=True):
    prefix = area_id_to_prefix(area_id)
    logger.info("prep_image_mask for {}".format(prefix))
    if is_valtrain:
        fn_list = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)
    else:
        fn_list = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)

    df = pd.read_csv(fn_list, index_col='ImageId')
    df_summary = load_train_summary_data(area_id)
    logger.info("Prepare image container: {}".format(fn_mask))
    with tb.open_file(fn_mask, 'w') as f:
        for image_id in tqdm.tqdm(df.index, total=len(df)):
            for pos, im_mask in get_slice_mask_im(df_summary, image_id):
                atom = tb.Atom.from_dtype(im_mask.dtype)
                slice_id = image_id + "_" + str(pos)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, slice_id, atom,
                                     im_mask.shape,
                                     filters=filters)
                ds[:] = im_mask


def prep_mul_image_store_train(area_id, datapath, is_valtrain=True):
    prefix = area_id_to_prefix(area_id)
    bandstats_mul = __load_mul_bandstats(area_id)

    logger.info("prep_mul_image_store_train for ".format(prefix))
    if is_valtrain:
        fn_list = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTRAIN_MUL_STORE.format(prefix)
    else:
        fn_list = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = FMT_VALTEST_MUL_STORE.format(prefix)

    df_list = pd.read_csv(fn_list, index_col='ImageId')

    logger.info("Image store file: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
            for slice_pos, im in get_slice_8chan_im(image_id,
                                                    datapath,
                                                    bandstats_mul):
                slice_id = '{}_{}'.format(image_id, slice_pos)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, slice_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im


def prep_mul_image_store_test(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    bandstats_mul = __load_mul_bandstats(area_id)

    logger.info("prep_mul_image_store_test for ".format(prefix))
    fn_list = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    fn_store = FMT_TEST_MUL_STORE.format(prefix)

    df_list = pd.read_csv(fn_list, index_col='ImageId')

    logger.info("Image store file: {}".format(fn_store))
    with tb.open_file(fn_store, 'w') as f:
        for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
            for slice_pos, im in get_slice_8chan_im(image_id,
                                                    datapath,
                                                    bandstats_mul,
                                                    is_test=True):
                slice_id = '{}_{}'.format(image_id, slice_pos)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, slice_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im


def prep_valtrain_test_slice_image(area_id):
    prefix = area_id_to_prefix(area_id)
    logger.info("prep_valtrain_test_slice_image for {}".format(prefix))

    df_train = pd.read_csv(
        FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
        index_col='ImageId')
    df_test = pd.read_csv(
        FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
        index_col='ImageId')
    band_cut_th = __load_band_cut_th(
        FMT_RGB_BANDCUT_TH_PATH.format(prefix))[area_id]
    df_summary = load_train_summary_data(area_id)

    fn = FMT_VALTRAIN_IM_STORE.format(prefix)
    logger.info("Prepare image container: {}".format(fn))
    if not Path(fn).exists():
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                for slice_pos, im in get_slice_3chan_im(image_id, band_cut_th):
                    slice_id = image_id + "_{}".format(slice_pos)
                    atom = tb.Atom.from_dtype(im.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root, slice_id, atom, im.shape,
                                         filters=filters)
                    ds[:] = im

    fn = FMT_VALTEST_IM_STORE.format(prefix)
    logger.info("Prepare image container: {}".format(fn))
    if not Path(fn).exists():
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                for slice_pos, im in get_slice_3chan_im(image_id, band_cut_th):
                    slice_id = image_id + "_{}".format(slice_pos)
                    atom = tb.Atom.from_dtype(im.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root, slice_id, atom, im.shape,
                                         filters=filters)
                    ds[:] = im

    fn = FMT_VALTEST_MASK_STORE.format(prefix)
    logger.info("Prepare image container: {}".format(fn))
    if not Path(fn).exists():
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                for pos, im_mask in get_slice_mask_im(df_summary, image_id):
                    atom = tb.Atom.from_dtype(im_mask.dtype)
                    slice_id = image_id + "_" + str(pos)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root, slice_id, atom, im_mask.shape,
                                         filters=filters)
                    ds[:] = im_mask


def calc_mul_multiband_cut_threshold(area_id, datapath):
    rows = []
    band_cut_th = __calc_mul_multiband_cut_threshold(area_id, datapath)
    prefix = area_id_to_prefix(area_id)
    row = dict(prefix=area_id_to_prefix(area_id))
    row['area_id'] = area_id
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        FMT_MUL_BANDCUT_TH_PATH.format(prefix), index=False)


def __calc_mul_multiband_cut_threshold(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    band_values = {k: [] for k in range(8)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(8)}

    image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(8):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove censored mask
                band_values[i_chan].append(values_)

    image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(8):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove censored mask
                band_values[i_chan].append(values_)

    for i_chan in range(8):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    return band_cut_th


def get_test_image_path_from_imageid(image_id, datapath, mul=False):
    prefix = image_id_to_prefix(image_id)
    if mul:
        return FMT_TEST_MSPEC_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)
    else:
        return FMT_TEST_RGB_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)


def get_train_image_path_from_imageid(image_id, datapath, mul=False):
    prefix = image_id_to_prefix(image_id)
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


def prep_mulmean(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    X_train = []

    # Load valtrain
    fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
    image_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(image_list):
            slice_pos = 5
            slice_id = image_id + '_' + str(slice_pos)
            im = np.array(f.get_node('/' + slice_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_train.append(im)

    # Load valtest
    fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
    image_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(image_list):
            slice_pos = 5
            slice_id = image_id + '_' + str(slice_pos)
            im = np.array(f.get_node('/' + slice_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_train.append(im)

    X_mean = np.array(X_train).mean(axis=0)

    fn = FMT_MULMEAN.format(prefix)
    logger.info("Prepare mean image: {}".format(fn))
    with tb.open_file(fn, 'w') as f:
        atom = tb.Atom.from_dtype(X_mean.dtype)
        filters = tb.Filters(complib='blosc', complevel=9)
        ds = f.create_carray(f.root, 'mulmean', atom, X_mean.shape,
                             filters=filters)
        ds[:] = X_mean


# >>> -------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command()
@click.argument('datapath', type=str)
def preproc_train(datapath):
    """ train.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("Preproc for training on {}".format(prefix))

    # Working directory
    working_dir = Path(FMT_VALTRAIN_MASK_STORE.format(prefix)).parent
    if not working_dir.exists():
        working_dir.mkdir(parents=True)

    # Imagelist (from v5)
    assert Path(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).exists()
    assert Path(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).exists()

    # Band stats (MUL)
    assert Path(FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists()

    # Mask (Target output)
    if Path(FMT_VALTRAIN_MASK_STORE.format(prefix)).exists():
        logger.info("Generate MASK (valtrain) ... skip")
    else:
        logger.info("Generate MASK (valtrain)")
        prep_image_mask(area_id, is_valtrain=True)
    if Path(FMT_VALTEST_MASK_STORE.format(prefix)).exists():
        logger.info("Generate MASK (valtest) ... skip")
    else:
        logger.info("Generate MASK (valtest)")
        prep_image_mask(area_id, is_valtrain=False)

    # Image HDF5 store (MUL)
    if Path(FMT_VALTRAIN_MUL_STORE.format(prefix)).exists():
        logger.info("Generate MUL_STORE (valtrain) ... skip")
    else:
        logger.info("Generate MUL_STORE (valtrain)")
        prep_mul_image_store_train(area_id, datapath, is_valtrain=True)
    if Path(FMT_VALTEST_MUL_STORE.format(prefix)).exists():
        logger.info("Generate MUL_STORE (valtest) ... skip")
    else:
        logger.info("Generate MUL_STORE (valtest)")
        prep_mul_image_store_train(area_id, datapath, is_valtrain=False)

    # Image Mean (MUL)
    if Path(FMT_MULMEAN.format(prefix)).exists():
        logger.info("Generate MULMEAN ... skip")
    else:
        logger.info("Generate MULMEAN")
        prep_mulmean(area_id, datapath)

    # DONE!
    logger.info("Preproc for training on {} ... done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def preproc_test(datapath):
    """ test.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("preproc_test for {}".format(prefix))

    # Imagelist
    assert Path(FMT_TEST_IMAGELIST_PATH.format(
        prefix=prefix)).exists()

    # Image HDF5 store (MUL)
    if Path(FMT_TEST_MUL_STORE.format(prefix)).exists():
        logger.info("Generate MUL_STORE (test) ... skip")
    else:
        logger.info("Generate MUL_STORE (test)")
        prep_mul_image_store_test(area_id, datapath)

    logger.info("preproc_test for {} ... done".format(prefix))


if __name__ == '__main__':
    cli()
