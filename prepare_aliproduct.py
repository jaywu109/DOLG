import argparse
import pandas as pd
from metric.utils import find_image


def main():
    parser = argparse.ArgumentParser('Aliproduct dataset preparing')
    parser.add_argument('-i', '--input', required=True, help='path of the Aliproduct root directory')
    parser.add_argument('-o', '--output', default='out.csv', help='path of the output csv')
    args = parser.parse_args()

    paths, ids = [], []

    for folder, path in find_image(args.input):
        paths.append(path)
        ids.append(folder)

    df = pd.DataFrame.from_dict({'path': paths, 'id': ids})
    df.to_csv(args.output, index=None)


if __name__ == '__main__':
    main()
