# Importing Libraries
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from Utils.constants import PROCESSED_DIR,RESULTS_DIR
import joblib

# Load the dataset
input_file = os.path.join(PROCESSED_DIR, "Dynamic_pricing adjustment with new cost.csv")
df= pd.read_csv(input_file)


                                            # Data Spliting and Model training


x=df[['Number_of_Riders','Number_of_Drivers','Location_Category','Average_Ratings','Customer_Loyalty_Status',
      'Number_of_Past_Rides','Time_of_Booking','Vehicle_Type','Expected_Ride_Duration']] #Feature column

y=df['New_cost'] #Target column

X_train_val, X_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

loyalty_order = ['Regular', 'Silver', 'Gold']
vehicle_type_order = ['Economy', 'Premium']

transformer= ColumnTransformer(transformers=[
    # Apply OneHotEncoder for Nominal columns
        ('location', OneHotEncoder(), ['Location_Category']),
        ('time', OneHotEncoder(), ['Time_of_Booking']),
        
        # Apply OrdinalEncoder for Ordinal columns with specific orders
        ('loyalty', OrdinalEncoder(categories=[loyalty_order]), ['Customer_Loyalty_Status']),
        ('vehicle', OrdinalEncoder(categories=[vehicle_type_order]), ['Vehicle_Type']),
        
        # Apply StandardScaler for Numerical columns
        ('scaler', StandardScaler(), ['Average_Ratings','Number_of_Past_Rides','Expected_Ride_Duration'])
    ],
    remainder='passthrough'
)

transformer.fit(X_train)

# Transform the training, validation, and test sets
X_train_trns = transformer.transform(X_train)
X_val_trns = transformer.transform(X_val)
X_test_trns = transformer.transform(X_test)


#using Linearregression
lr=RandomForestRegressor()                                    
lr.fit(X_train_trns,y_train)


                                            # Model Evaluation & Check prediction

# 1. Model evaluation on Training Data
y_train_pred_lr = lr.predict(X_train_trns)
train_rmse_lr = np.sqrt(mean_squared_error(y_train,y_train_pred_lr))
train_r2_lr = r2_score(y_train,y_train_pred_lr)


# 2. Model evaluation on validation Data
y_val_pred_lr = lr.predict(X_val_trns)
val_rmse_lr = np.sqrt(mean_squared_error(y_val,y_val_pred_lr))
val_r2_lr = r2_score(y_val,y_val_pred_lr)


# 3. Model evaluation on Test Data
y_test_pred_lr = lr.predict(X_test_trns)
test_rmse_lr = np.sqrt(mean_squared_error(y_test, y_test_pred_lr))
test_r2_lr = r2_score(y_test, y_test_pred_lr)



                                            
                                            # Save Test data, Prediction and Error
results_df = pd.DataFrame(X_test, columns=x.columns)
results_df['Actual'] = y_test
results_df['Predicted_RandomForest'] = y_test_pred_lr
results_df['Error_RF'] = y_test_pred_lr-y_test
results_df['Predicted_LinearRegression'] = y_test_pred_lr
results_df['Error_LR'] = y_test_pred_lr-y_test
output_file = os.path.join(RESULTS_DIR, "test_results by model_5.csv")
results_df.to_csv(output_file, index=False)


joblib.dump(lr, "model.pkl")
joblib.dump(transformer, "transformer.pkl")
