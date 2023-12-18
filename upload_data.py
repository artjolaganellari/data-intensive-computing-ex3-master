from fileinput import filename
import os
import requests
import time
import sys


def upload(folder='images/test2017', remote_url='http://128.131.30.90:5040/upload?token=mytoken'):
    upload_times = list()
    for file in os.listdir(folder):
        start_time = time.time()
        with open(os.path.join(folder, file), 'rb') as file_handle:
            res = requests.post(remote_url, files={'file': file_handle})
        end_time = time.time()
        upload_times.append(end_time - start_time)
        sys.stderr.write(str(upload_times[-1]))
        sys.stderr.flush()

    print(upload_times)


if __name__ == '__main__':
    upload()
