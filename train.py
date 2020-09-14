import pandas as pd 
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import time
import logging 
import spectrum
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn import svm


#************************************
# pre-processing 
#************************************

class Processor:
    def __init__(self, base_path: str, train_data_path: str, 
                 test_data_path:str, chemical_path:str, 
                 task_path:str, file_number: int)-> None:
        
        self.base_path = base_path
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.chemical_path = chemical_path
        self.task_path = task_path
        self.file_number= file_number 
        self.train_folder= os.path.join(base_path, f"{task_path}{str(file_number)}", train_data_path)
        self.test_folder= os.path.join(base_path, f"{task_path}{str(file_number)}", test_data_path)

    def get_data(self)-> pd.DataFrame:
        spectrum_list= list()
        file_number = list()

        
        for file in os.listdir(self.train_folder):
            try:
                data = pd.read_csv(os.path.join(self.train_folder, file),
                                   header = None, 
                                   float_precision="high",
                                   index_col= None)
                
                spectrum_list.append(data.iloc[:,1].values)
                file_number.append(file.split(".")[0])
            
            except Exception as e:
                logging.info(str(e))
                pass 
            
        return pd.DataFrame({"file_number":file_number, "spectrum_data": spectrum_list})

    def get_chemicals(self, col1= None, col2= None)-> pd.DataFrame:
        data = pd.read_excel(os.path.join(self.base_path, f"{self.task_path}{str(self.file_number)}", self.chemical_path))
        
        if (col1 is None and col2 is None):
            return data
        return data[[col1, col2]]

    #display graph for wavelength and intensity value; 
    def plot_spectrum(self, flatten_file: pd.DataFrame)-> None:
        
        plt.figure(figsize=(8, 6), dpi=100)
        for i in range(len(flatten_file.index)):
            plt.xlabel("Wavelength") #波长
            plt.ylabel("Absorbance (AU)") #吸光度
            plt.plot(flatten_file["spectrum_data"][i])
            plt.grid(True)
            plt.tight_layout()
            
    #reprocess data to row to apply SVM model
    def flatten2row(self, path)-> pd.DataFrame:
        spectrum_data= list()
        file_name = list()
        for file in os.listdir(path):
            try:
                data = (pd.read_csv(os.path.join(path,file),
                                    header= None, 
                                    index_col= None))
            
                spectrum_data.append(data.iloc[:,1].values)
                file_name.append(file.split(".")[0])
            
            except Exception as e:
                pass 
            
        return pd.DataFrame(spectrum_data), file_name

    def get_traintest(self, traintest):
        if traintest == "TRAIN":
            return self.flatten2row(self.train_folder)
        
        return self.flatten2row(self.test_folder)

    
    def merge_df(self, unique_col, col1= None, col2= None):
        #add sample_number column to match with chemical file
        train_data, file_name = self.get_traintest("TRAIN") 
        train_data[unique_col]= file_name
        train_data[unique_col]= train_data[unique_col].apply(lambda x: int(x))
        if col1 is None and col2 is None:
            final_full_df= self.get_chemicals().merge(train_data, on=unique_col, how= "left")
            
        chemicals_file= self.get_chemicals(col1, col2).rename(columns={col1:unique_col})
        chemicals_file[unique_col]= chemicals_file[unique_col].apply(lambda x: int(str(x).split("-")[-1]))
        final_full_df= chemicals_file.merge(train_data, on=unique_col, how= "left")

        return final_full_df

#************************************
# baseline model - SVM
#************************************
class Classifier:
    def svm(self, train_data, test_data, train_y):
        cls = svm.SVC(kernel="linear")
        #train
        train_data.fillna("0", inplace= True)
        train_y.fillna("0", inplace= True)
        
        cls.fit(train_data, train_y)
        #predict
        pred = cls.predict(test_data)
        return pred 

    def knn(self):
        pass 

# no model inference, no ground truth label provided



if __name__ == "__main__":
    
    BASE_PATH= "data/数据挖掘题目"
    TRAIN_DATA_PATH= "建模集光谱"
    TEST_DATA_PATH ="验证集光谱"
    CHEMICAL_EXCEL_PATH= "建模集化学值.xlsx"
    TASK_PATH ="近红外试题" #base for three different files paths

    processor = Processor(BASE_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, CHEMICAL_EXCEL_PATH, TASK_PATH, 1)
    train_data, file_name= processor.get_traintest("TRAIN")
    test_data, _= processor.get_traintest("TEST")
    merge_df= processor.merge_df("样本序号") # "样本序号"
    train_x = merge_df.iloc[:, 4:]
    train_y = merge_df["等级"]
    #change train_y => 样本序号
    train_y_2 = merge_df["样本序号"]
    
    model= Classifier()
    pred= model.svm(train_x, test_data, train_y)
    pred2= model.svm(train_x, test_data, train_y_2)
    pd.DataFrame({"等级预测":pred}).to_csv("predication/task1_pred.csv", index= False)
    # pred2

    #************************************
    ## pred for task 2
    
    #************************************
    ## pred for task 3 
    
    processor3 = Processor(BASE_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH, CHEMICAL_EXCEL_PATH, TASK_PATH, 3)
    train_data3, file_name3 = processor3.get_traintest("TRAIN")
    test_data3, _= processor3.get_traintest("TEST")
    #merge
    merge_df3= processor3.merge_df("样本序号", "Sample", "化学值")
    train_x_3= merge_df3.iloc[:, 2:]
    train_y_3= merge_df3["样本序号"]
    pred3= model.svm(train_x_3, test_data3, train_y_3)
    pd.DataFrame({"样本序号预测": pred3}).to_csv("predication/task3_pred.csv", index= False)

    