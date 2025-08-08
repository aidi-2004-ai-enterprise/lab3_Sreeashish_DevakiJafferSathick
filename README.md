README.md
Penguin Species Classification API
A machine learning API that predicts penguin species (Adelie, Chinstrap, Gentoo) based on physical measurements using XGBoost and FastAPI.
🚀 Quick Start
Local Setup
bash# Install dependencies
uv add fastapi uvicorn pandas numpy scikit-learn xgboost pydantic

# Train model
python train.py

# Run API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Docker
bashdocker build -t penguin-api .
docker run -p 8080:8080 penguin-api
📡 API Endpoints

GET / - Basic info
GET /health - Health check
POST /predict - Species prediction
GET /docs - Interactive docs

Example Prediction
bashcurl -X POST "https://penguin-api-194534203106.us-central1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181.0,
    "body_mass_g": 3750.0,
    "sex": "male",
    "island": "Torgersen"
  }'
🔗 Live Service

URL: https://penguin-api-194534203106.us-central1.run.app
Docs: https://penguin-api-194534203106.us-central1.run.app/docs

📊 Model Performance

Training Accuracy: 100%
Test Accuracy: 100%
F1-Score: 1.0

🧪 Testing
bashuv add pytest pytest-cov httpx
uv run pytest tests/test_api.py -v --cov=app.main
📁 Project Structure
├── app/
│   ├── main.py           # FastAPI app
│   └── data/             # Model files
├── tests/
│   └── test_api.py       # Unit tests
├── Dockerfile            # Container config
├── requirements.txt      # Dependencies
└── train.py             # Model training
Assignment Questions
What edge cases might break your model in production that aren't in your training data?

Extreme measurements (bill length > 70mm, body mass > 10kg)
Measurements from penguin chicks or elderly penguins
Data from new geographic locations not in training set
Seasonal variations in penguin measurements
Injured or sick penguins with abnormal measurements

What happens if your model file becomes corrupted?

API returns 500 error on startup
Health check endpoint reports model not loaded
All prediction requests fail with detailed error messages
Need to re-upload model from backup or retrain

What's a realistic load for a penguin classification service?

Research institutions: 100-1000 requests/day
Educational demos: 10-100 requests/day
Wildlife monitoring: 1000-10000 requests/day during breeding season
Peak load estimate: 50 concurrent users maximum

How would you optimize if response times are too slow?

Cache model in memory (currently loads from GCS each time)
Increase Cloud Run CPU/memory allocation
Use model quantization to reduce inference time
Implement request batching for multiple predictions
Add response caching for identical inputs

What metrics matter most for ML inference APIs?

Response time (95th percentile < 500ms)
Prediction accuracy (monitor for model drift)
Error rate (< 1% for valid inputs)
Throughput (requests per second)
Model confidence scores distribution

Why is Docker layer caching important for build speed?

Dependencies don't change often, so they're cached
Only rebuild layers that changed (app code)
Reduces build time from 5+ minutes to 30 seconds
Our Dockerfile installs requirements before copying code

What security risks exist with running containers as root?

Container breakout could compromise host system
Processes inside container have unnecessary privileges
File system access beyond what's needed
Our solution: Created non-root user 'appuser'

How does cloud auto-scaling affect your load test results?

Cold starts add latency to initial requests
New instances take time to provision (5-10 seconds)
First few requests hit existing warm instances
Load tests show higher latency during scale-up events

What would happen with 10x more traffic?

Cloud Run would scale to max instances (10)
Response times would increase due to resource contention
Might hit quota limits for concurrent requests
Need to increase max instances and monitor costs

How would you monitor performance in production?

Google Cloud Monitoring for basic metrics
Custom metrics for prediction accuracy
Error rate alerting
Response time dashboards
Log analysis for unusual patterns

How would you implement blue-green deployment?

Deploy new version to separate Cloud Run service
Test new version with small traffic percentage
Gradually shift traffic using Cloud Run traffic splitting
Keep old version running until new version proven stable

What would you do if deployment fails in production?

Immediately rollback to previous working version
Check Cloud Run logs for error details
Verify all environment variables are set correctly
Test deployment in staging environment first
Have monitoring alerts for deployment failures

What happens if your container uses too much memory?

Cloud Run kills the container with OOM error
Service becomes unavailable until new instance starts
Need to monitor memory usage and optimize model loading
Increase memory allocation or optimize code