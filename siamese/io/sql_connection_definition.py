import pandas as pd
import sqlalchemy as db
from sqlalchemy.engine.url import URL

from siamese.io.path_definition import get_project_dir
from siamese.settings import MYSQL_APPAREL_DATABASE_CONFIG, MYSQL_SHOE_DATABASE_CONFIG

def set_db_connection(fashion: str) -> URL:

    # %% CHECK24 Credentials

    database_config = eval(f'MYSQL_{fashion}_DATABASE_CONFIG')
    db_connect_live_slave = URL(database_config['drivername'],
                                host=database_config['host'],
                                port=database_config['port'],
                                username=database_config['username'],
                                password=database_config['password'])

    return db_connect_live_slave


def load_image(fashion: str) -> pd.DataFrame:
    """

    Args:
        fashion: fashion type for the server

    Returns:
        A dataframe containing image paths and the corresponding metadata
    """

    sql_query = """
                SELECT  p.id as 'inventory_product_id',
                        va.variation_id,
                        mo.category_id,    
                        IFNULL(JSON_UNQUOTE(JSON_EXTRACT(JSON_MERGE_PATCH(JSON_OBJECTAGG(IFNULL(pa.name, ''), pa.value), JSON_OBJECTAGG(IFNULL(va.name, ''), va.value), JSON_OBJECTAGG(IFNULL(ma.name, ''), ma.value)), '$.brand')), '') AS manufacturer_brand
                FROM backofficeinventory_products.product p 
                JOIN backofficeinventory_products.variation v ON (p.variation_id = v.id)
                JOIN backofficeinventory_products.model mo ON (v.model_id = mo.id)
                LEFT JOIN backofficeinventory_products.attribute pa ON (pa.product_id = p.id)
                LEFT JOIN backofficeinventory_products.attribute va ON (va.variation_id = v.id)
                LEFT JOIN backofficeinventory_products.attribute ma ON (ma.model_id = mo.id)
                group by p.id
                """

    db_connect_live_slave = set_db_connection(fashion)
    engine = db.create_engine(db_connect_live_slave)

    with engine.connect() as connection:
        db2_manufacturer = pd.read_sql(sql_query, connection)

    db2_manufacturer = db2_manufacturer.dropna(how='any').drop(labels=['product_id'], axis=1).\
        drop_duplicates(subset=['variation_id'])

    sql_query = """
                SELECT m.variation_id, m.legacy_cdn_url, m.view_angle
                FROM backofficeinventory_products.media m
                WHERE m.view_angle in ('side_r', 'front_r', 'side_l', 'front_l', 'back_r', 'back_l')
                """

    with engine.connect() as connection:
        db2_media = pd.read_sql(sql_query, connection)

    db2_media = db2_media.merge(db2_manufacturer, on='variation_id')

    # since we are going to use straitified process for sampling, drop all images without brand and category

    db2_media.dropna(subset=['category_id', 'manufacturer_brand'], inplace=True)

    return db2_media.merge(db2_manufacturer, on='variation_id')


if __name__ == "__main__":

    db2 = load_image(fashion='SHOE')

    db2.to_csv(f"{get_project_dir()}/data/train/shoe_image_index.csv")



