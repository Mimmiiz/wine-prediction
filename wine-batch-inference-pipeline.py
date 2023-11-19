import os
import modal

LOCAL=False

if LOCAL == False:
    stub = modal.Stub("wine_batch_inference")
    hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe-image"])
    @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
    def f():
        g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")

    feature_view = fs.get_feature_view(name="wine", version =1)
    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    offset = 1
    wine_quality = y_pred[y_pred.size-offset]

    print(f"Predicted Wine Quality: {wine_quality}")
    
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read()

    todays_wine = df.iloc[-offset]
    todays_wine["quality"] = wine_quality
    todays_wine_df = todays_wine.to_frame().T
    todays_wine_df.rename(columns={"quality": "pred_quality"}, inplace=True)

    if todays_wine_df['type'].iloc[0] == 0:
        todays_wine_df.loc[:, 'type'] = 'White'
    else:
        todays_wine_df.loc[:, 'type'] = 'Red'
    todays_wine_df.reset_index(drop=True, inplace=True)

    dfi.export(todays_wine_df.T, './latest_wine.png', table_conversion = 'matplotlib')
    dataset_api = project.get_dataset_api()
    image_resources = "Resources/wine_images" 
    dataset_api.upload("./latest_wine.png", image_resources, overwrite=True)

    label = df.iloc[-offset]["quality"]
    print(f"Actual Wine Quality: {label}")
    actual_wine = df.iloc[-offset]
    actual_wine_df = actual_wine.to_frame().T

    if actual_wine_df['type'].iloc[0] == 0:
        actual_wine_df.loc[:, 'type'] = 'White'
    else:
        actual_wine_df.loc[:, 'type'] = 'Red'
    actual_wine_df.reset_index(drop=True, inplace=True)

    dfi.export(actual_wine_df.T, './actual_wine.png', table_conversion = 'matplotlib')
    dataset_api.upload("./actual_wine.png", image_resources, overwrite=True)

    monitor_fg = fs.get_or_create_feature_group(
        name="wine_predictions",
        version=1,
        primary_key=["datetime"],
        description="Wine quality prediction/Outcome monitoring"
    )

    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine_quality],
        'label': [label],
        'datetime': [now]
    }

    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", image_resources, overwrite=True)

    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    num_predictions = predictions.value_counts().count()
    print(f"Number of different wine quality predictions to date: {num_predictions}")
    if num_predictions == 8:
        results = confusion_matrix(labels, predictions)

        df_cm = pd.DataFrame(results, ["True 3", "True 4", "True 5", "True 6", "True 7", "True 8", "True 9"], 
                     ["Pred 3", "Pred 4", "Pred 5", "Pred 6", "Pred 7", "Pred 8", "Pred 9"])
        
        cm = sns.heatmap(df_cm, annot=True, fmt='.0f')
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", image_resources, overwrite=True)
    else:
        print("You need 8 different wine quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 8 different wine quality predictions") 

if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        with stub.run():
            f()

