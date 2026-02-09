# AWS Cost Estimate

Rough monthly estimate for 2,000 client pipelines on AWS + Snowflake.

## Assumptions

- 2,000 client runs per month, concentrated on Day 1
- ~65% model reuse rate (700 retrain, 1,300 reuse)
- 3 AWS regions (data residency)
- Spot instances for all batch workloads

---

## Monthly Breakdown

| Service | What | Monthly Cost |
|---------|------|-------------|
| **Snowflake** | Data prep (multi-cluster S warehouse, ~233 cluster-hrs) + splits (~67 cluster-hrs) + ad-hoc | **$950** |
| **MWAA** | Small environment base (~$350) + Day 1 burst workers (~$100) | **$450** |
| **Ray Cluster** | Day 1: 20 nodes/region x 3 regions x 8h, spot pricing | **$75** |
| **Networking** | NAT Gateways (3 regions) + cross-region metadata | **$100** |
| **MLflow** | ECS Fargate + RDS PostgreSQL (small) + S3 artifacts | **$85** |
| **SageMaker** | 700 training jobs on ml.m5.xlarge spot, ~15 min each | **$25** |
| **Monitoring** | CloudWatch logs/metrics + SNS alerts | **$35** |
| **S3 Storage** | Models (~35GB) + simulation results (~200GB) + staging | **$8** |
| **TOTAL** | | **~$1,730** |

**Cost per client run: ~$0.87**

---

## Key notes

**Snowflake (55% of total)** is the dominant cost. We keep the S warehouse (already handles 3-11 min per client) and use multi-cluster mode for concurrency instead of upsizing to L. This avoids over-provisioning compute per query. Clustering keys on the date column and per-second billing + auto-suspend keep costs tight.

**Ray scales to zero on Days 2-31.** Since simulations only run on Day 1, the cluster spins up, processes, and terminates. Using EKS with Karpenter or bare EC2 spot fleets avoids always-on cost. The $75 estimate reflects Day 1 burst only with spot pricing.

**MWAA is the fixed-cost "tax."** The ~$350 base environment runs 24/7. For lower cost, Step Functions (~$50/month for 2,000 executions) is an alternative, but loses the Airflow UI and Python-native DAG logic that makes branching/retrain decisions clean.

**Model reuse saves time, not money.** SageMaker training is already cheap ($25/month). The real value of skipping 1,300 training jobs is throughput: each skipped job frees 7-23 minutes of pipeline time on Day 1.

---

## Model Reuse Impact

| Scenario | Training Jobs | Training Cost | Pipeline Time Saved |
|----------|--------------|---------------|---------------------|
| Always retrain | 2,000 | ~$70 | Baseline |
| 65% reuse | 700 | ~$25 | ~325 hrs of training skipped |
| 80% reuse | 400 | ~$15 | ~470 hrs of training skipped |
