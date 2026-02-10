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

## Model Reuse Impact

| Scenario | Training Jobs | Training Cost | Pipeline Time Saved |
|----------|--------------|---------------|---------------------|
| Always retrain | 2,000 | ~$70 | Baseline |
| 65% reuse | 700 | ~$25 | ~325 hrs of training skipped |
| 80% reuse | 400 | ~$15 | ~470 hrs of training skipped |
