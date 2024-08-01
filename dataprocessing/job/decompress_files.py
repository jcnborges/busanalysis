from datetime import timedelta
import lzma
import shutil
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

sys.path.insert(0, os.path.abspath('..'))

parser = ArgumentParser()
parser.add_argument("-s", "--start-date", dest="start_date", help="start date", metavar="DATE")
parser.add_argument("-e", "--end-date", dest="end_date", help="end date", metavar="DATE")
parser.add_argument("-fd", "--folder", dest="folder", help="folder", metavar="FOLDER")
parser.add_argument("-fl", "--file", dest="file", help="file", metavar="FILE")

parser.add_argument("-d", "--delete", dest="delete", help="delete staging files", metavar="DELETE")

args = parser.parse_args()

start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
folder = args.folder
file = args.file
delete = args.delete


def decompress_files(folder, file, start_date, end_date):
    print(f"Date range: {start_date} --> {end_date}")
    print(f"Filename: {file}")

    delta = end_date - start_date
    print(delta)

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        download_file_day = day.strftime("%Y_%m_%d")

        datareferencia = day.replace(day=1).strftime("%Y-%m")

        base_folder = f"/data/raw/{datareferencia}/{folder}"

        os.makedirs(base_folder, exist_ok=True)

        try:
            fstaging = f"/data/staging/{datareferencia}/{folder}/{download_file_day}_{file}"
            fraw = f"{base_folder}/{download_file_day}_{file.replace('.xz', '')}"

            binary_data_buffer = lzma.open(fstaging, mode='rt', encoding='utf-8').read()

            with open(fraw, 'w') as a:
                a.write(binary_data_buffer)
            print(f"{fraw} decompressed")
        except Exception as err:
            print(f"Can't open file: {file} for date {download_file_day}")


def delete_files():
    base_folder = f"/data/staging/"
    shutil.rmtree(base_folder, ignore_errors=True)


decompress_files(folder, file, start_date, end_date)

if delete == 'y':
    delete_files()
