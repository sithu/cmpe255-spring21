import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt


def prepare_X(df):
        base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        df = df.copy()
        features = base.copy()
        df['age'] = 2020 - df.year
        features.append('age')

        for v in [2, 3, 4]:
            feature = 'num_doors_%s' % v
            df[feature] = (df['number_of_doors'] == v).astype(int)
            features.append(feature)

        for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
            feature = 'is_make_%s' % v
            df[feature] = (df['make'] == v).astype(int)
            features.append(feature)

        for v in ['regular_unleaded', 'premium_unleaded_(required)', 
                'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
            feature = 'is_type_%s' % v
            df[feature] = (df['engine_fuel_type'] == v).astype(int)
            features.append(feature)

        for v in ['automatic', 'manual', 'automated_manual']:
            feature = 'is_transmission_%s' % v
            df[feature] = (df['transmission_type'] == v).astype(int)
            features.append(feature)

        for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
            feature = 'is_driven_wheens_%s' % v
            df[feature] = (df['driven_wheels'] == v).astype(int)
            features.append(feature)

        for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
            feature = 'is_mc_%s' % v
            df[feature] = (df['market_category'] == v).astype(int)
            features.append(feature)

        for v in ['compact', 'midsize', 'large']:
            feature = 'is_size_%s' % v
            df[feature] = (df['vehicle_size'] == v).astype(int)
            features.append(feature)

        for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
            feature = 'is_style_%s' % v
            df[feature] = (df['vehicle_style'] == v).astype(int)
            features.append(feature)

        df_num = df[features]
        df_num = df_num.fillna(0)
        X = df_num.values
        return X

def calculate_rmse(r):
    carprice=CarPrice()
    w_0,w = carprice.reg_train(r)
    rmse_value = carprice.rmse(y_val, test(w_0, w,df_val))
    return rmse_value

class CarPrice:

    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
    
     
    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')
   
    def validate(self):
        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)
        idx = np.arange(n)
        np.random.shuffle(idx)
        df_shuffled = self.df.iloc[idx]

        df_train = df_shuffled.iloc[:n_train].copy()
        df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        df_test = df_shuffled.iloc[n_train+n_val:].copy()

        y_train_orig = df_train.msrp.values
        y_val_orig = df_val.msrp.values
        y_test_orig = df_test.msrp.values
       

        y_train = np.log1p(y_train_orig)
        y_val = np.log1p(y_val_orig)
        y_test = np.log1p(y_test_orig)

        del df_train['msrp']
        del df_val['msrp']
        del df_test['msrp']
        
        return df_train,df_val,df_test,y_train,y_val,y_test

    def linear_regression(self, X, y,r):
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        reg = r * np.eye(XTX.shape[0])
        XTX = XTX + reg
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(y)
        return w[0], w[1:]

    def rmse(self, y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
   
    def reg_train(self, r):
        X_train = prepare_X(df_train)
        return self.linear_regression(X_train, y_train, r)

    def show_results(self, X, y, y_pred):
        columns = ['engine_cylinders','transmission_type','driven_wheels','number_of_doors',
                   'market_category','vehicle_size','vehicle_style','highway_mpg','city_mpg','popularity']
        X = X.copy()
        X = X[columns]
        X['msrp'] =np.expm1(y.round(2))
        X['msrp_pred'] =np.expm1 (y_pred.round(2))
        print(X.head(5).to_string(index=False))
    

def test(w_0, w, X):
        carprice=CarPrice()
        X = prepare_X(X)
        return w_0 + X.dot(w)

if __name__ == '__main__':
    carprice=CarPrice()
    carprice.trim()
    df_train,df_val,df_test,y_train,y_val,y_test= carprice.validate()
    r_min= min([0, 0.0001, 0.001, 0.01, 0.1, 0.7, 1, 10, 100], key=calculate_rmse)
    print("best value of reg parameter is:",r_min)
    w0,w=carprice.reg_train(r_min)
    y_pred=test(w0,w,df_test)
    print("Original msrp vs. Predicted msrp of 5 cars") 
    carprice.show_results(df_test,y_test,y_pred)
    print("RMSE :",carprice.rmse(y_test,y_pred))


