import os, time
import numpy as np
import subprocess
import pandas as pd

rootdir = os.path.split(__file__)[0]


def gpcc_encode(filedir, bin_dir, show=False):

    subp = subprocess.Popen(rootdir + '/tmc3' +
                            ' --mode=0' +
                            ' --positionQuantizationScale=1' +
                            ' --trisoupNodeSizeLog2=0' +
                            ' --neighbourAvailBoundaryLog2=8' +
                            ' --intra_pred_max_node_size_log2=6' +
                            ' --inferredDirectCodingMode=0' +
                            ' --maxNumQtBtBeforeOt=4' +
                            ' --uncompressedDataPath=' + filedir +
                            ' --compressedStreamPath=' + bin_dir,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        c = subp.stdout.readline()

    return


def gpcc_decode(bin_dir, rec_dir, show=False):
    subp = subprocess.Popen(rootdir + '/tmc3' +
                            ' --mode=1' +
                            ' --compressedStreamPath=' + bin_dir +
                            ' --reconstructedDataPath=' + rec_dir +
                            ' --outputBinaryPly=0'
                            ,
                            shell=True, stdout=subprocess.PIPE)
    c = subp.stdout.readline()
    while c:
        if show: print(c)
        c = subp.stdout.readline()

    return


def get_points_number(filedir):
    plyfile = open(filedir)

    line = plyfile.readline()
    while line.find("element vertex") == -1:
        line = plyfile.readline()
    number = int(line.split(' ')[-1][:-1])

    return number


def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try:
            number = float(item)
        except ValueError:
            continue

    return number


def pc_error(infile1, infile2, res, normal=False, show=False):
    
    headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)",
                "h.       1(p2point)", "h.,PSNR  1(p2point)"]

    headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)",
                "h.       2(p2point)", "h.,PSNR  2(p2point)"]

    headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)",
                "h.        (p2point)", "h.,PSNR   (p2point)"]

    haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                      "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                      "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

    headers = headers1 + headers2 + headersF + haders_p2plane

    command = str(rootdir + '/pc_error_d' +
                  ' -a ' + infile1 +
                  ' -b ' + infile2 +
                  ' --hausdorff=1 ' +
                  ' --resolution=' + str(res - 1))

    if normal:
        headers += haders_p2plane
        command = str(command + ' -n ' + infile1)

    results = {}

    start = time.time()
    subp = subprocess.Popen(command,
                            shell=True, stdout=subprocess.PIPE)

    c = subp.stdout.readline()
    while c:
        line = c.decode(encoding='utf-8')  # python3.
        if show:
            print(line)
        for _, key in enumerate(headers):
            if line.find(key) != -1:
                value = number_in_line(line)
                results[key] = value

        c = subp.stdout.readline()
        
    return pd.DataFrame([results])

