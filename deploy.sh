#!/bin/bash

# Deployment script for Penguin Classification API
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="penguin-api"
REPO_NAME="penguin-repo"
IMAGE_NAME="penguin-api"

echo -e "${GREEN}Starting deployment of Penguin Classification API...${NC}"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Function to print status
print_status() {
    echo -e "${YELLOW}$1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if gcloud is authenticated
print_status "Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
    exit 1
fi
print_success "Authenticated with gcloud"

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    print_error "No project set. Please run 'gcloud config set project PROJECT_ID'"
    exit 1
fi
print_success "Project ID: $PROJECT_ID"

# Enable required APIs
print_status "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com \
                      run.googleapis.com \
                      artifactregistry.googleapis.com \
                      storage.googleapis.com \
                      --quiet

print_success "APIs enabled"

# Create Artifact Registry repository if it doesn't exist
print_status "Setting up Artifact Registry..."
if ! gcloud artifacts repositories describe $REPO_NAME --location=$REGION &>/dev/null; then
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REGION \
        --description="Docker repository for penguin API"
    print_success "Artifact Registry repository created"
else
    print_success "Artifact Registry repository already exists"
fi

# Configure Docker authentication
print_status "Configuring Docker authentication..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
print_success "Docker authentication configured"

# Build the image
print_status "Building Docker image..."
docker build --platform linux/amd64 -t $IMAGE_NAME .
print_success "Docker image built"

# Tag the image
print_status "Tagging image for registry..."
docker tag $IMAGE_NAME ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest
docker tag $IMAGE_NAME ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$(date +%Y%m%d-%H%M%S)
print_success "Image tagged"

# Push to Artifact Registry
print_status "Pushing image to Artifact Registry..."
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest
docker push ${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$(date +%Y%m%d-%H%M%S)
print_success "Image pushed to registry"

# Create service account if it doesn't exist
print_status "Setting up service account..."
SA_NAME="penguin-api-sa"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe $SA_EMAIL &>/dev/null; then
    gcloud iam service-accounts create $SA_NAME \
        --display-name="Penguin API Service Account"
    
    # Add necessary roles
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="roles/storage.objectViewer"
    
    print_success "Service account created and configured"
else
    print_success "Service account already exists"
fi

# Create GCS bucket if it doesn't exist
print_status "Setting up Cloud Storage bucket..."
BUCKET_NAME="penguin-model-artifacts-$PROJECT_ID"
if ! gsutil ls gs://$BUCKET_NAME &>/dev/null; then
    gsutil mb gs://$BUCKET_NAME
    print_success "Cloud Storage bucket created"
else
    print_success "Cloud Storage bucket already exists"
fi

# Upload model artifacts to GCS (optional)
print_status "Uploading model artifacts to GCS..."
if [ -f "app/data/model.json" ]; then
    gsutil cp app/data/model.json gs://$BUCKET_NAME/
    gsutil cp app/data/label_encoder.pkl gs://$BUCKET_NAME/
    gsutil cp app/data/metadata.json gs://$BUCKET_NAME/
    print_success "Model artifacts uploaded to GCS"
else
    print_error "Model artifacts not found. Please run training script first."
    exit 1
fi

# Deploy to Cloud Run
print_status "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=${REGION}-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:latest \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --port=8080 \
    --memory=2Gi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=10 \
    --timeout=300 \
    --service-account=$SA_EMAIL \
    --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME,ENVIRONMENT=production" \
    --quiet

print_success "Deployment completed!"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform=managed \
    --region=$REGION \
    --format="value(status.url)")

echo -e "${GREEN}"
echo "=================================================="
echo "🚀 Deployment Successful!"
echo "=================================================="
echo "Service URL: $SERVICE_URL"
echo "Health Check: $SERVICE_URL/health"
echo "API Docs: $SERVICE_URL/docs"
echo "=================================================="
echo -e "${NC}"

# Test the deployment
print_status "Testing deployment..."
if curl -f -s "$SERVICE_URL/health" > /dev/null; then
    print_success "Health check passed"
    
    # Test prediction endpoint
    print_status "Testing prediction endpoint..."
    PREDICTION_RESULT=$(curl -s -X POST "$SERVICE_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }')
    
    if echo "$PREDICTION_RESULT" | grep -q "predicted_species"; then
        print_success "Prediction endpoint working"
        echo "Sample prediction: $(echo $PREDICTION_RESULT | jq -r .predicted_species)"
    else
        print_error "Prediction endpoint failed"
    fi
else
    print_error "Health check failed"
    echo "Please check the logs: gcloud run logs tail $SERVICE_NAME --region=$REGION"
fi

echo -e "${YELLOW}Next steps:${NC}"
echo "1. Monitor your service: https://console.cloud.google.com/run"
echo "2. View logs: gcloud run logs tail $SERVICE_NAME --region=$REGION"
echo "3. Run load tests: locust -f locustfile.py --host $SERVICE_URL"
echo "4. Set up monitoring and alerting"