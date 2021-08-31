import hashlib

from sklearn.model_selection import GroupKFold

from siamese.io.sql_connection_definition import load_image

def hash_process(string: str) -> str:

    h1.update(string.encode(encoding='utf-8'))

    return h1.hexdigest()

if __name__ == "__main__":

    df = load_image('SHOE')

    df['brand_category'] = df[['category_id', 'manufacturer_brand']].agg("_".join, axis=1)

    h1 = hashlib.sha1()

    df['bc_hash'] = df['brand_category'].apply(lambda x: hash_process(x))

    gkf = GroupKFold(n_splits=3)  # 3 because train/val/test three folds

    count = 0

    for train_index, _ in gkf.split(df, groups=df['bc_hash'].values):
        if count == 0:
            df_train = df.loc[train_index]
            df_sum = df.groupby('variation_id').count()
        elif count == 1:
            df_val = df.loc[train_index]
        else:
            df_test = df.loc[train_index]

        count += 1



