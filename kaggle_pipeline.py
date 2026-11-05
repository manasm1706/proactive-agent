"""
Kafka + Spark Structured Streaming demonstration.
Run with a Kafka broker already running and the Spark Kafka connector available.
For the easiest local setup, use the included docker-compose.yml.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import pandas as pd
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp, window, sum as spark_sum, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="data/transaction_data.csv")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="bank-transactions")
    parser.add_argument("--output", default="outputs/streaming")
    parser.add_argument("--limit", type=int, default=0, help="Optional event limit; 0 means all rows")
    parser.add_argument("--rate", type=float, default=0.02, help="Seconds between produced events")
    args = parser.parse_args()
    path = Path(args.transactions)
    if not path.exists(): raise FileNotFoundError(f"Cannot find {path}. Put Kaggle transaction_data.csv in data/.")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ","_") for c in df.columns]
    required = {"customer_id","transaction_date","transaction_amount"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; found {list(df.columns)}")
    if "investment_amount" not in df: df["investment_amount"] = 0.0
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    df = df.dropna(subset=["transaction_date","transaction_amount","customer_id"]).sort_values("transaction_date")
    if args.limit: df = df.head(args.limit)
    producer = KafkaProducer(bootstrap_servers=args.bootstrap,
        value_serializer=lambda x: json.dumps(x, default=str).encode("utf-8"),
        key_serializer=lambda x: str(x).encode("utf-8"))
    sent = 0
    for _, r in df.iterrows():
        event = {"customer_id":str(r["customer_id"]), "transaction_date":r["transaction_date"].isoformat(),
                 "transaction_amount":float(r["transaction_amount"]),
                 "investment_amount":float(pd.to_numeric(r.get("investment_amount",0), errors="coerce") or 0)}
        producer.send(args.topic, key=event["customer_id"], value=event)
        sent += 1
        if args.rate: time.sleep(args.rate)
    producer.flush()
    producer.close()
    print(f"Produced {sent} historical transaction events to {args.topic}.")
    schema = StructType([
        StructField("customer_id",StringType()),StructField("transaction_date",StringType()),
        StructField("transaction_amount",DoubleType()),StructField("investment_amount",DoubleType())])
    spark = (SparkSession.builder.appName("ProactiveDigitalEngagement")
             .config("spark.sql.shuffle.partitions","4")
             .config("spark.jars.packages","org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1").getOrCreate())
    raw = (spark.readStream.format("kafka").option("kafka.bootstrap.servers",args.bootstrap)
           .option("subscribe",args.topic).option("startingOffsets","earliest").load())
    events = (raw.select(from_json(col("value").cast("string"),schema).alias("e"))
              .select("e.*").withColumn("event_time",to_timestamp("transaction_date")))
    agg = (events.withWatermark("event_time","1 day")
            .groupBy(window(col("event_time"), "30 days"), col("customer_id"))
            .agg(count("*").alias("transaction_count"),
                 spark_sum("transaction_amount").alias("transaction_value"),
                 spark_sum("investment_amount").alias("investment_value")))
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    query = (agg.writeStream.outputMode("append").format("parquet")
             .option("path",str(output/"monthly_customer_metrics"))
             .option("checkpointLocation",str(output/"checkpoint"))
             .trigger(availableNow=True).start())
    query.awaitTermination()
    spark.stop()
    print(f"Spark Structured Streaming output written to {output/'monthly_customer_metrics'}")

if __name__ == "__main__": main()
