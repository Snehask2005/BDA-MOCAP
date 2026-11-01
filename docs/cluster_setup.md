# Cluster setup, data and real-bill validation (owner: Anaswara)

This is the runbook for the multi-node experiments (Phases 5 and 6). Everything
in `execution/` and `experiments/` also runs on a laptop with synthetic data; the
steps below are only needed for the paper's SF10 / SF100 numbers and the real
cloud bill.

> Prices and machine types change. Check the provider's current price page for
> your region and update `config/cloud_pricing.json` before reporting costs.

## 1. Generate TPC-DS data

```bash
pip install duckdb
python scripts/generate_tpcds.py --sf 10     # benchmarks/tpcds/sf10/parquet + 99 queries
python scripts/generate_tpcds.py --sf 100    # on a large VM (~150 GB free disk)
```

Upload to the cluster's object store:

```bash
# Dataproc
gsutil -m cp -r benchmarks/tpcds/sf10/parquet  gs://<bucket>/tpcds/sf10/parquet
gsutil -m cp -r benchmarks/tpcds/sf100/parquet gs://<bucket>/tpcds/sf100/parquet
# EMR
aws s3 sync benchmarks/tpcds/sf10/parquet  s3://<bucket>/tpcds/sf10/parquet
aws s3 sync benchmarks/tpcds/sf100/parquet s3://<bucket>/tpcds/sf100/parquet
```

Then point `data.tpcds_sf10` / `data.tpcds_sf100` in `configs/spark_env.yaml`
at those `gs://` or `s3://` paths.

## 2. Create a labelled cluster

Label every cluster with the batch it runs (`mocap-batch=<experiment>`), so the
bill can be split per batch later. One batch = one ephemeral cluster.

**Dataproc (GCP)**

```bash
gcloud dataproc clusters create mocap-e1 \
  --region=us-central1 --image-version=2.3-debian12 \
  --master-machine-type=n2-standard-4 \
  --worker-machine-type=n2-standard-8 --num-workers=4 \
  --labels=mocap-batch=e1_main \
  --properties=spark:spark.ui.enabled=true \
  --max-idle=30m
```

Check that the image ships Spark 4.x (`gcloud dataproc clusters describe ...`);
if not, use the newest image or install Spark 4.2.0 via an initialization action.

**EMR (AWS)**

```bash
aws emr create-cluster --name mocap-e1 --release-label <emr-release-with-spark-4> \
  --applications Name=Spark \
  --instance-groups InstanceGroupType=MASTER,InstanceCount=1,InstanceType=m5.xlarge \
                    InstanceGroupType=CORE,InstanceCount=4,InstanceType=m5.2xlarge \
  --tags mocap-batch=e1_main --use-default-roles --auto-terminate
```

Update the `cluster` profile in `configs/spark_env.yaml` (executor instances,
cores, memory) to match the machines you chose.

## 3. Ship the code and run

```bash
git clone https://github.com/Snehask2005/BDA-MOCAP.git && cd BDA-MOCAP
pip install -r requirements.txt
export MOCAP_PROFILE=cluster

spark-submit experiments/e1_main.py --data tpcds_sf10 --tag tpcds_sf10 --repeats 10
python experiments/e2_budget.py     --data tpcds_sf10 --tag tpcds_sf10      # reuses E1 measurements
python experiments/e5_case_study.py --data tpcds_sf10 --tag tpcds_sf10

# official TPC-DS query text instead of the star-join workload
spark-submit experiments/e1_main.py --data tpcds_sf10 --sql-dir benchmarks/tpcds/queries --tag tpcds99_sf10
```

Queries the candidate generator cannot handle (e.g. outer joins, sub-queries
it does not rewrite) are listed under `queries_skipped` in `e1_run_info.json`.
Report that coverage in the paper.

Robustness (Phase 6) on real data:

```bash
spark-submit experiments/robustness.py --data tpcds_sf10 --tag tpcds_sf10 \
  --shift-data gs://<bucket>/tpcds/sf100/parquet \
  --stale-workdir gs://<bucket>/mocap/stale_work \
  --skew-data <path to a skewed variant, or drop zipf_skew from --scenarios>
```

Copy results back before the cluster is deleted:

```bash
gsutil -m cp -r results gs://<bucket>/mocap-results/$(date +%F)/
```

## 4. Real cloud bill

1. Enable billing export: GCP → Billing → *Billing export* → BigQuery;
   AWS → Cost Explorer with the `mocap-batch` cost-allocation tag activated.
2. Wait for costs to appear (often 24 h or more).
3. Export cost per batch to CSV with columns `batch,cost`. On GCP:

```sql
SELECT l.value AS batch, SUM(cost) AS cost
FROM `<project>.<dataset>.gcp_billing_export_v1_<id>`, UNNEST(labels) AS l
WHERE l.key = 'mocap-batch'
GROUP BY batch
```

4. Reconcile and draw `paper/figures/real_cost.pdf`:

```bash
python experiments/cloud_cost.py --provider dataproc --instance n2-standard-8 \
  --telemetry "results/telemetry/*.jsonl" \
  --measurements results/phase5/tpcds_sf10/measurements \
  --bill bills/dataproc_by_batch.csv
```

Cluster bills include idle time, the master node and storage, so billed totals
are higher than telemetry-derived executor cost. The paper claim is about
**rank correlation** across batches, not equality.

## 5. Protocol checklist (from the reviewer audit)

- [ ] `ANALYZE TABLE` run before measuring (the scripts do this unless `--no-analyze`)
- [ ] join reordering off and broadcast threshold `-1` for candidate runs; `plan_verified` true in telemetry
- [ ] 10–30 repeats per plan; warm and cold cache runs recorded (`cache_mode`)
- [ ] budgets relative to the Oracle (1.0×, 1.2×, 1.5×, 2.0×)
- [ ] medians with 95% CIs and Wilcoxon tests; losses reported
- [ ] cluster shape, Spark version and config committed with the results (`*_run_info.json`)
