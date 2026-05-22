# test_fridge.py

from fridge import load, encoding

dataset = load("../dataset2/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv", operation="u")

print("DATA:")
print(dataset.data)

print("\nTARGET:")
print(dataset.target)

encoded = encoding(dataset, num_bits=8)

print("\nENCODED:")

for item in encoded:
    print(item)