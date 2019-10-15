# Implementation of DAE feature engineering funcs
denoising autoencoder based feature engineering was used in [Porto Seguro’s 1st solution](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629)

# how to use example
```
from get_dae_feature import preprocess, train_dae, get_representation

# 読み込み
train_data = pd.read_csv(DATA_PATH+"sample_train.csv")
test_data = pd.read_csv(DATA_PATH+"sample_test.csv")
train_data["train"] = 1
test_data["train"] = 0
train_y = train_data["charges"].values

num_cols = ["bmi", "age"]
cat_cols = ["children", "region", "smoker", "sex"]

# dae用にmerge
merged_data = pd.concat([train_data, test_data], sort=False)
merged_data = merged_data.drop(["id", "charges"], axis=1)
merged_data["children"] = merged_data["children"].map(lambda x: str(x))

# rankgauss & one hot & swap noise
one_hot_df = preprocess(merged_data, num_cols, cat_cols)

# train, test data
train_x = one_hot_df.query("train == 1").drop("train", axis=1)
test_x = one_hot_df.query("train == 0").drop("train", axis=1)

train_dae(one_hot_df.drop("train", axis=1), "cuda:9", "models/dae.model", cycle=1000)

train_x = get_representation(train_x.values, "models/dae.model")
test_x =  get_representation(test_x.values, "models/dae.model")

train_data_ = pd.DataFrame(train_x)
train_data_["charges"] = train_y
```