def train_predict(historical_data:list,
                      historical_column_names:list,
                      forecast_data:list,
                      forecast_column_names:list):
    
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import numpy as np

    #Read in the data
    historical_data=np.array(historical_data)
    objects=[pd.DataFrame(historical_data[i:i+1]) for i in range(0,22)]
    tbl=pd.concat(objects, axis=1)
    #Change the column names to what our table in Snowflake has
    tbl.columns=historical_column_names

    # #Split into training and validation
    X_train=tbl.drop(columns=['CHURNVALUE','CUSTOMERID'])
    y_train=tbl['CHURNVALUE']
    #return np.array(X_train)
    # setup pipeline

    #transformations
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import FunctionTransformer

    #Classifier
    from sklearn.ensemble import RandomForestClassifier

    #Pipeline
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split

    #Model Accuracy
    from sklearn.metrics import balanced_accuracy_score

    # Model Pipeline
    ord_pipe = make_pipeline(
        FunctionTransformer(lambda x: x.astype(str)) ,
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        )

    num_pipe = make_pipeline(
        SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0),
        MinMaxScaler()
        )

    clf = make_pipeline(RandomForestClassifier(random_state=0, n_jobs=-1))

    model = make_pipeline(ord_pipe, num_pipe, clf)

    # fit the model
    model.fit(X_train, y_train)
    
    # predict 
    #forecast = pd.DataFrame(forecast_data, forecast_column_names)
    forecast_data=np.array(forecast_data)
    objects=[pd.DataFrame(forecast_data[i:i+1]) for i in range(0,22)]
    tb2=pd.concat(objects, axis=1)
    #Change the column names to what our table in Snowflake has
    tb2.columns=forecast_column_names

    # #Split into training and validation
    X_Predict=tb2.drop(columns=['CHURNVALUE','CUSTOMERID'])
    return model.predict(X_Predict)
