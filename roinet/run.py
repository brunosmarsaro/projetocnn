import csv
import os

def make_dic(mias_list):
    mias_dic = {}
    for img in mias_list:
        mias_dic[img[0]] = img[4:]
    return mias_dic 

if __name__ == "__main__":
    dirimg_path = "./database/achado"
    mias_list = []
    with open("mias_info.csv","r") as mias_info:
        r_mias = csv.reader(mias_info, delimiter=" ")
        for row in r_mias:
            if len(row) != 7: continue
            mias_list.append(row)

    mias_dic = make_dic(mias_list)

    for file in os.listdir(dirimg_path):
        file_name, file_ext = os.path.splitext(file)
        
