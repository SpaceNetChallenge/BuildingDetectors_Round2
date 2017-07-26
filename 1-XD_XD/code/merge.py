# -*- coding: utf-8 -*-
from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
import sys
import warnings


MODEL_NAME = 'v17'
MODEL_DIR = "/data/working/models/{}".format(MODEL_NAME)
FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"

LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'


# Logger
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter(LOGFORMAT))
logger = getLogger('spacenet2')
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)


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


def _merge(area_id_list, output_fn):
    # file check
    for area_id in area_id_list:
        prefix = area_id_to_prefix(area_id)
        fn_out = FMT_TESTPOLY_PATH.format(prefix)
        if not Path(fn_out).exists():
            logger.info("Required file not found: {}".format(fn_out))
            sys.exit(1)

    # merge files
    rows = []
    for area_id in area_id_list:
        prefix = area_id_to_prefix(area_id)
        fn_out = FMT_TESTPOLY_PATH.format(prefix)
        with open(fn_out, 'r') as f:
            line = f.readline()
            if area_id == area_id_list[0]:
                rows.append(line)  # header line
            for line in f:
                line = _remove_interiors(line)
                rows.append(line)
    with open(output_fn, 'w') as f:
        for line in rows:
            f.write(line)


def merge():
    if len(sys.argv) < 3:
        print("Usage: merge.py [/data/test/AOI_2_Vegas_Test ...] out.csv")

    test_path_list = sys.argv[1:-1]
    output_fn = sys.argv[-1]

    if not output_fn.endswith('.csv'):
        print("Error: output must be end with '.csv'.")
        sys.exit(1)

    area_id_list = [
        directory_name_to_area_id(datapath)
        for datapath in test_path_list]
    _merge(area_id_list, output_fn)


if __name__ == '__main__':
    merge()
