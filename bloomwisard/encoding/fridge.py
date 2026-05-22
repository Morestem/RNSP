import numpy as np
import os
import re
import sys
from bitarray import bitarray
from thermometer import Thermometer

class Fridge:

    def __init__(self):

        self.data = []
        self.target = []

        self.condition_map = {
            "low": 0,
            "high": 1
        }

    def encode(self, row):

        columns = row.strip().split(",")

        # ignorar cabeçalho
        if columns[0] == "date":
            return

        # remover espaços
        columns = [c.strip() for c in columns]

        temperature = float(columns[2])

        condition = self.condition_map[columns[3]]

        label = int(columns[4])
        #print(f"Temperature: {temperature}, Condition: {condition}, Label: {label}")
        #breakpoint()
        self.data.append(
            np.array([temperature, condition], dtype=np.float64)
        )

        self.target.append(label)

def load(filename, operation = None):
    basename = os.path.basename(filename)
    cfiledata = basename + ".data"
    cfiletarget = basename + ".target"

    if operation == "u" or not (os.path.isfile(cfiledata + ".npy") or os.path.isfile(cfiletarget + ".npy")): #Update .npy files
        fp = open(filename, "r")
        fridge = Fridge()

        try:
            for line in fp:
                if re.search(r"\d", line):
                    fridge.encode(line)


            fridge.data = np.ndarray(shape=(len(fridge.data), 2), buffer=np.array(fridge.data, dtype=np.float64), dtype=np.float64)
            fridge.target = np.ndarray(shape=(len(fridge.target),), buffer=np.array(fridge.target, dtype=np.int64), dtype=np.int64)
        finally:
            fp.close()
            np.save(cfiledata, fridge.data)
            np.save(cfiletarget, fridge.target)
    else:
        fridge = Fridge()
        fridge.data = np.load(cfiledata + ".npy")
        fridge.target = np.load(cfiletarget + ".npy")

    return fridge

def encoding(dataset, num_bits=8):

        inputs = []

        th = Thermometer(
            dataset.data.min(),
            dataset.data.max(),
            num_bits
        )

        for i in range(len(dataset.data)):

            row = dataset.data[i]

            bits = bitarray()

            # temperatura
            bits.extend(th.code(row[0]))

            # condição high/low
            bits.append(int(row[1]))

            inputs.append({
                "class": dataset.target[i],
                "bitarray": bits
            })

        return inputs