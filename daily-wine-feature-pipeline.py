import modal

LOCAL=False

if LOCAL== False:
    stub = modal.Stub("daily_wine")
    image = modal.Image.debian_slim().pip_install(["hopsworks"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
    def f():
        g()

def generate_wine(quality, type_std, type_mean, volatile_acidity_std, volatile_acidity_mean, 
                  citric_acid_std, citric_acid_mean, chlorides_std, chlorides_mean,
                  density_std, density_mean, sulphates_std, sulphates_mean, alcohol_std, alcohol_mean):
    """
    Returns a wine as a single row in a DataFrame
    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        'type': [round(np.random.normal(type_mean, type_std))],
        'volatile_acidity': [np.random.normal(volatile_acidity_mean, volatile_acidity_std)],
        'citric_acid': [np.random.normal(citric_acid_mean, citric_acid_std)],
        'chlorides': [np.random.normal(chlorides_mean, chlorides_std)],
        'density': [np.random.normal(density_mean, density_std)],
        'sulphates': [np.random.normal(sulphates_mean, sulphates_std)],
        'alcohol': [np.random.normal(alcohol_mean, alcohol_std)],
    })

    df['quality'] = quality
    return df

def get_wine_of_quality(df, quality):
    df_new = df[df['quality'] == quality]

    std = df_new.std()
    mean = df_new.mean()

    new_wine = generate_wine(quality, 
                             std["type"], mean["type"], 
                             std["volatile_acidity"], mean["volatile_acidity"],
                             std["citric_acid"], mean["citric_acid"], 
                             std["chlorides"], mean["chlorides"],
                             std["density"], mean["density"], 
                             std["sulphates"], mean["sulphates"],
                             std["alcohol"], mean["alcohol"])
    
    return new_wine


def get_random_wine(df):
    """
    Returns a DataFrame containg a random wine
    """
    import pandas as pd
    import random

    wine_quality_3 = get_wine_of_quality(df, 3)
    wine_quality_4 = get_wine_of_quality(df, 4)
    wine_quality_5 = get_wine_of_quality(df, 5)
    wine_quality_6 = get_wine_of_quality(df, 6)
    wine_quality_7 = get_wine_of_quality(df, 7)
    wine_quality_8 = get_wine_of_quality(df, 8)
    wine_quality_9 = get_wine_of_quality(df, 9)

    # randomly pick one of these 9 and write it to the featurestore
    pick_random = random.randint(3, 9)

    if pick_random == 3:
        wine_df = wine_quality_3
        print("Wine of quality 3")
    elif pick_random == 4:
        wine_df = wine_quality_4
        print("Wine of quality 4")
    elif pick_random == 5:
        wine_df = wine_quality_5
        print("Wine of quality 5")
    elif pick_random == 6:
        wine_df = wine_quality_6
        print("Wine of quality 6")
    elif pick_random == 7:
        wine_df = wine_quality_7
        print("Wine of quality 7")
    elif pick_random == 8:
        wine_df = wine_quality_8
        print("Wine of quality 8")
    else:
        wine_df = wine_quality_9
        print("Wine of quality 9")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine", version=1)
    query = wine_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name="wine_all", 
        version=1, 
        query=query
    )

    df = feature_view.get_batch_data()
    # print(df)

    wine_df = get_random_wine(df)
    # print(wine_df)

    wine_fg = fs.get_feature_group(name="wine", version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()


