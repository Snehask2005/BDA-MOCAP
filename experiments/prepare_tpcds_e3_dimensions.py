from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructField, StructType, StringType


RAW = Path("benchmarks/tpcds/raw/sf10")
OUT = Path("benchmarks/tpcds/e3/parquet")

# Read the source columns in their original TPC-DS order.
# Store only the join keys and attributes needed by E3 workloads.
TABLES = {
    "customer": {
        "file": "customer_1_8.dat",
        "columns": """
            c_customer_sk c_customer_id c_current_cdemo_sk
            c_current_hdemo_sk c_current_addr_sk
            c_first_shipto_date_sk c_first_sales_date_sk
            c_salutation c_first_name c_last_name
            c_preferred_cust_flag c_birth_day c_birth_month
            c_birth_year c_birth_country c_login c_email_address
            c_last_review_date_sk
        """.split(),
        "key": "c_customer_sk",
        "integer_columns": [
            "c_customer_sk",
            "c_current_addr_sk",
            "c_current_hdemo_sk",
        ],
        "keep": [
            "c_customer_sk",
            "c_first_name",
            "c_current_addr_sk",
            "c_current_hdemo_sk",
        ],
    },
    "customer_address": {
        "file": "customer_address_1_8.dat",
        "columns": """
            ca_address_sk ca_address_id ca_street_number
            ca_street_name ca_street_type ca_suite_number ca_city
            ca_county ca_state ca_zip ca_country ca_gmt_offset
            ca_location_type
        """.split(),
        "key": "ca_address_sk",
        "keep": ["ca_address_sk", "ca_state"],
    },
    "household_demographics": {
        "file": "household_demographics_1_8.dat",
        "columns": """
            hd_demo_sk hd_income_band_sk hd_buy_potential
            hd_dep_count hd_vehicle_count
        """.split(),
        "key": "hd_demo_sk",
        "keep": ["hd_demo_sk", "hd_buy_potential"],
    },
    "time_dim": {
        "file": "time_dim_1_8.dat",
        "columns": """
            t_time_sk t_time_id t_time t_hour t_minute t_second
            t_am_pm t_shift t_sub_shift t_meal_time
        """.split(),
        "key": "t_time_sk",
        "keep": ["t_time_sk", "t_hour"],
    },
    "promotion": {
        "file": "promotion_1_8.dat",
        "columns": """
            p_promo_sk p_promo_id p_start_date_sk p_end_date_sk
            p_item_sk p_cost p_response_target p_promo_name
            p_channel_dmail p_channel_email p_channel_catalog
            p_channel_tv p_channel_radio p_channel_press
            p_channel_event p_channel_demo p_channel_details
            p_purpose p_discount_active
        """.split(),
        "key": "p_promo_sk",
        "keep": ["p_promo_sk", "p_channel_email"],
    },
}


def main() -> None:
    for table, spec in TABLES.items():
        source = RAW / spec["file"]
        if not source.is_file():
            raise FileNotFoundError(f"Missing source file: {source}")

        if len(spec["columns"]) != len(set(spec["columns"])):
            raise ValueError(f"Duplicate source columns for {table}")

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("MOCAP-E3-Dimension-Preparation")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        for table, spec in TABLES.items():
            source = RAW / spec["file"]
            destination = OUT / table

            schema = StructType([
                StructField(column, StringType(), True)
                for column in spec["columns"]
            ])

            df = (
                spark.read
                .option("sep", "|")
                .option("header", "false")
                .schema(schema)
                .csv(source.resolve().as_uri())
            )

            selected = []
            for name in spec["keep"]:
                column = F.col(name)
                integer_columns = set(
                    spec.get("integer_columns", [spec["key"]])
                )
                if name in integer_columns:
                    column = column.cast("int")
                selected.append(column.alias(name))

            # Overwrite only this table under the isolated E3 directory.
            df.select(*selected).write.mode("overwrite").parquet(
                destination.resolve().as_uri()
            )

            print(f"Prepared {table}: {destination}")

        print(f"\nE3 dimension preparation complete: {OUT}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
