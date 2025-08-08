Deployment Documentation
Project Info

Service URL: https://penguin-api-194534203106.us-central1.run.app
Project ID: seventh-capsule-468315-e3
Region: us-central1

Commands Used
1. Setup
powershell$PROJECT_ID = "seventh-capsule-468315-e3"
$REGION = "us-central1"
$BUCKET_NAME = "penguin-model-bucket-$PROJECT_ID"
$SERVICE_NAME = "penguin-api"
2. Create Service Account
powershellgcloud iam service-accounts create penguin-api-sa
gcloud projects add-iam-policy-binding $PROJECT_ID --member="serviceAccount:penguin-api-sa@$PROJECT_ID.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
gcloud iam service-accounts keys create sa-key.json --iam-account="penguin-api-sa@$PROJECT_ID.iam.gserviceaccount.com"
3. Upload Model to GCS
powershellgsutil cp app/data/model.json gs://$BUCKET_NAME/
gsutil cp app/data/label_encoder.pkl gs://$BUCKET_NAME/
gsutil cp app/data/metadata.json gs://$BUCKET_NAME/
4. Docker Build & Test
powershelldocker build --platform linux/amd64 -t $SERVICE_NAME .
docker run -d -p 8080:8080 --name test $SERVICE_NAME
curl http://localhost:8080/health
5. Artifact Registry
powershellgcloud artifacts repositories create penguin-repo --repository-format=docker --location=$REGION
gcloud auth configure-docker us-central1-docker.pkg.dev
docker tag $SERVICE_NAME us-central1-docker.pkg.dev/$PROJECT_ID/penguin-repo/$SERVICE_NAME:latest
docker push us-central1-docker.pkg.dev/$PROJECT_ID/penguin-repo/$SERVICE_NAME:latest
6. Deploy to Cloud Run
powershellgcloud run deploy $SERVICE_NAME --image=us-central1-docker.pkg.dev/$PROJECT_ID/penguin-repo/$SERVICE_NAME:latest --platform=managed --region=$REGION --allow-unauthenticated --port=8080 --memory=2Gi --cpu=1 --service-account=penguin-api-sa@$PROJECT_ID.iam.gserviceaccount.com --set-env-vars="GCS_BUCKET_NAME=$BUCKET_NAME,GCS_MODEL_BLOB=model.json,GCS_ENCODER_BLOB=label_encoder.pkl,GCS_METADATA_BLOB=metadata.json"
Issues & Solutions
Issue 1: Docker not running
Error: error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping"
Solution: Started Docker Desktop
Issue 2: Missing httpx dependency
Error: RuntimeError: The starlette.testclient module requires the httpx package
Solution: uv add httpx
Issue 3: PowerShell vs bash commands
Error: export : The term 'export' is not recognized
Solution: Used PowerShell syntax $VAR = "value"
Issue 4: Wrong logs command
Error: Invalid choice: 'logs'
Solution: Used gcloud run services logs read
Final Configuration

Memory: 2GB
CPU: 1 core
Max instances: 10
Service account: penguin-api-sa
Authentication: Unauthenticated (for demo)

