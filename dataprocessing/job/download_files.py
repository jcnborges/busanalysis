from datetime import timedelta
import urllib3

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

args = parser.parse_args()

start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
folder = args.folder
file = args.file


def download_files(folder, file, start_date, end_date, base_url):
    print(f"Date range: {start_date} --> {end_date}")
    print(f"Base URL: {base_url}")
    print(f"Filename: {file}")

    delta = end_date - start_date
    print(delta)

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        download_file_day = day.strftime("%Y_%m_%d")

        datareferencia = day.replace(day=1).strftime("%Y-%m")

        url = f"{base_url}{download_file_day}_{file}"
        print(f"Downloading: {url}")

        base_folder = f"/data/staging/{datareferencia}/{folder}"

        os.makedirs(base_folder, exist_ok=True)

        fd = f"/data/staging/{datareferencia}/{folder}/{download_file_day}_{file}"

        http = urllib3.PoolManager()
        r = http.request('GET', url, preload_content=False)

        with open(fd, 'wb') as out:
            while True:
                data = r.read()
                if not data:
                    break
                out.write(data)

        r.release_conn()
        print(f"Downloaded {url}")

    return "download realizado com sucesso"


download_files(folder, file, start_date, end_date, base_url="http://dadosabertos.c3sl.ufpr.br/curitibaurbs/")
