# Proactive Digital Engagement Agent

Big Data mini project for financial services using the public [Banking and Customer Transaction Data](https://www.kaggle.com/datasets/yogeshtekawade/banking-and-customer-transaction-data) dataset.

## What the project demonstrates

- **Dataset-driven analytics:** loads the three Kaggle CSV files: `customer_data.csv`, `transaction_data.csv`, and `bank_data.csv`.
- **Apache Kafka:** replays timestamp-sorted historical transactions as events to a Kafka topic.
- **Spark Structured Streaming:** consumes Kafka records using Spark's Kafka source and writes customer/month aggregates.
- **PySpark:** distributed aggregation of transaction count, transaction value and investment value by customer and month.
- **Customer behaviour analytics:** calculates transaction frequency, cumulative transaction amount, investment amount, average balance and recency.
- **Unsupervised analytics:** K-Means customer segmentation and Isolation Forest customer-level anomaly screening.
- **Recommendation system:** transparent heuristic ranking of illustrative financial products using available customer activity and a configurable relevance threshold.
- **Ten-agent workflow:** ingestion, behaviour analysis, pattern detection, potential-needs inference, decision, product recommendation, message generation, channel/timing, fatigue check and engagement monitoring.
- **Streamlit dashboard:** customer overview, agent pipeline, product catalogue, live customer test, analytics and data explorer.

## Dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/yogeshtekawade/banking-and-customer-transaction-data

Place the extracted files in the repository's `data/` directory:

```text
data/
├── customer_data.csv
├── transaction_data.csv
└── bank_data.csv
```

The code expects the dataset's documented columns, including:

- Customers: `Customer_ID`, `Age`, `Customer_Type`, `City`, `Region`
- Transactions: `Transaction_ID`, `Customer_ID`, `Account_Type`, `Total_Balance`, `Transaction_Amount`, `Investment_Amount`, `Investment_Type`, `Transaction_Date`
- Bank branches: `Branch_ID`, `City`, `Region`, `Firm_Revenue`, `Expenses`, `Profit_Margin`

Column names are normalised automatically for case and spaces. Customer and transaction files are required; bank data is optional.

## A. Run the Streamlit dashboard locally

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Upload the three CSV files in the sidebar, or place them in `data/` before starting the app.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## B. Run Kafka + Spark Structured Streaming locally

Prerequisites: Docker Desktop (or Docker Engine), Java compatible with Spark 3.5, Python dependencies installed, and the Kaggle CSVs in `data/`.

1. Start the single-node Kafka KRaft broker:

   ```bash
   docker compose up -d
   ```

2. Install the project requirements:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:

   ```bash
   python kaggle_pipeline.py --transactions data/transaction_data.csv --bootstrap localhost:9092
   ```

4. The script replays historical transactions into the `bank-transactions` Kafka topic. Spark consumes those records through the Kafka source and writes monthly customer metrics to `outputs/streaming/monthly_customer_metrics/`.

To run a smaller test:

```bash
python kaggle_pipeline.py --limit 1000 --rate 0.001
```

Stop Kafka:

```bash
docker compose down
```

### Spark Kafka connector

The script requests the matching Spark Kafka connector (`spark-sql-kafka-0-10_2.12:3.5.1`) through Spark's package resolver. Internet access is needed on the first run. If package resolution is blocked, start the script with the connector package explicitly, for example:

```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 kaggle_pipeline.py
```

For environments where `spark-submit` cannot execute the script directly, add the connector to `PYSPARK_SUBMIT_ARGS` before starting Python:

```bash
# Linux/macOS
export PYSPARK_SUBMIT_ARGS="--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 pyspark-shell"
python kaggle_pipeline.py
```

## C. Run the Big Data pipeline in Kaggle Notebooks

1. Create a Kaggle Notebook and attach the Banking and Customer Transaction Data dataset using **Add Input**.
2. Enable Internet if you need to install packages or download the Spark Kafka connector.
3. Upload or clone this repository into the notebook working directory.
4. Copy the attached Kaggle CSV files into a `data/` directory, or pass their full Kaggle input paths using `--transactions`.
5. Start a Kafka KRaft broker in the notebook session, or use an accessible Kafka broker. Kafka does not run automatically just because the notebook is open.
6. Install `pyspark==3.5.1`, `kafka-python` and the matching Spark Kafka connector.
7. Run `kaggle_pipeline.py` with the broker address and transaction file path.

Kaggle sessions are temporary. The broker and streaming query exist only while the session is running.

## Modelling and interpretation

- The selected dataset has no documented campaign-response label. Therefore, the project does **not** train or report a supervised customer-conversion model.
- K-Means is used for exploratory customer segmentation based on transaction count, transaction amount, investment amount, average balance and recency.
- Isolation Forest is used to flag unusual customer-level activity profiles. An anomaly flag is a review signal, not proof of fraud.
- Recommendation scores are transparent heuristic relevance scores, not calibrated probabilities or proof of financial suitability.
- A configurable default threshold of 50% suppresses low-scoring recommendations.
- Potential life-event language is treated as a hypothesis inferred from observed patterns, never as a verified fact.
- The dashboard's notification channel is simulated. No actual SMS, email, push notification or bank action is sent.
- The fatigue agent explains that a true notification-frequency check requires a historical message-send log, which is not supplied by the selected dataset.

## Ten-agent responsibilities

1. Data Ingestion Agent – validates customer and transaction records.
2. Behaviour Analysis Agent – inspects customer-level financial activity.
3. Pattern Detection Agent – identifies rule-based spending and activity signals.
4. Life Event Detection Agent – flags possible needs as hypotheses, not confirmed events.
5. Decision Agent – determines whether the engagement workflow should proceed.
6. Product Recommendation Agent – ranks products using transparent relevance rules.
7. Message Generation Agent – creates an informational personalised message.
8. Channel & Timing Agent – selects a simulated channel and engagement window.
9. Fatigue Detection Agent – checks notification history availability and suppresses where appropriate.
10. Engagement Monitoring Agent – records the simulated workflow outcome; real response tracking requires an integration and outcome data.

## Recommendation method

The baseline recommender scores a small illustrative catalogue using the customer's preferred investment type, investment activity, balance and transaction activity. Products are ranked by heuristic relevance and only candidates meeting the configured threshold are shown. This is an interpretable prototype, not a trained financial-advice model.

## Repository structure

```text
proactive-digital-engagement-agent/
├── app.py
├── agents.py
├── data_utils.py
├── modeling.py
├── kaggle_pipeline.py
├── requirements.txt
├── docker-compose.yml
├── README.md
├── .gitignore
└── data/
    └── .gitkeep
```

## Dataset attribution

Dataset: Yogesh Tekawade, *Banking and Customer Transaction Data*, Kaggle.

https://www.kaggle.com/datasets/yogeshtekawade/banking-and-customer-transaction-data

The dataset is described as synthetic banking, customer and transaction data. Review the Kaggle dataset page for the current licence and attribution requirements before redistribution. This repository does not include the raw dataset files.
