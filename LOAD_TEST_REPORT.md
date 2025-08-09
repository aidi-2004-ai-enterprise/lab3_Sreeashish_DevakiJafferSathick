Load Test Report
Test Setup

Target: https://penguin-api-194534203106.us-central1.run.app
Tool: Locust
Date: 2025-08-08

Test Results
Baseline Test (1 user, 60s)
bashuv run locust -f locustfile.py --host $SERVICE_URL --users 1 --spawn-rate 1 --run-time 60s --headless
Results:

Total requests: 29
RPS: 0.50
Avg response time: 89 ms
Failures: 0%

Normal Load (10 users, 5min)
bashuv run locust -f locustfile.py --host $SERVICE_URL --users 10 --spawn-rate 2 --run-time 300s --headless
Results:

Total requests:  1,458
RPS: 4.87
Avg response time: 72 ms
Failures: 0%

Stress Test (50 users, 2min)
bashuv run locust -f locustfile.py --host $SERVICE_URL --users 50 --spawn-rate 5 --run-time 120s --headless
Results:

Total requests: 2,845
RPS: 23.75
Avg response time: 72 ms
Failures: 0%

Analysis
Performance

Cold starts add ~2-3 seconds to first requests
Warm instances respond in ~100-300ms
Auto-scaling works but has delay

Bottlenecks

Model loading from GCS on cold start
Container startup time
XGBoost inference time

Cloud Run Behavior

Scales up when CPU > 60%
Takes 5-10 seconds to provision new instances
Scales down after 15 minutes idle

Recommendations
Performance

Set min instances to 1 to avoid cold starts
Cache model in memory instead of loading from GCS
Increase CPU allocation for faster inference

Cost Optimization

Keep min instances at 0 for cost savings
Monitor actual memory usage and reduce if possible
Set up billing alerts

Production

Add authentication
Implement rate limiting
Set up monitoring alerts
Add response caching
