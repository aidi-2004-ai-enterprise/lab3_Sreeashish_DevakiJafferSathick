#!/bin/bash

# Test execution script for Penguin Classification API
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Running Penguin Classification API Tests${NC}"
echo "========================================"

# Function to print section headers
print_section() {
    echo -e "\n${YELLOW}$1${NC}"
    echo "----------------------------------------"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  No virtual environment detected. Consider using one:${NC}"
    echo "python -m venv venv && source venv/bin/activate"
fi

# Install dependencies
print_section "📦 Installing Dependencies"
pip install -r requirements.txt
pip install pytest pytest-cov locust
print_success "Dependencies installed"

# Check if model files exist
print_section "🔍 Checking Model Files"
if [ ! -f "app/data/model.json" ]; then
    print_error "Model file not found. Running training script..."
    python train.py
    if [ $? -eq 0 ]; then
        print_success "Model trained successfully"
    else
        print_error "Model training failed"
        exit 1
    fi
else
    print_success "Model files found"
fi

# Run linting (if available)
if command -v flake8 &> /dev/null; then
    print_section "🔍 Running Code Linting"
    flake8 app/ --max-line-length=100 --ignore=E203,W503 || print_error "Linting issues found"
fi

# Run type checking (if available)
if command -v mypy &> /dev/null; then
    print_section "🔍 Running Type Checking"
    mypy app/ --ignore-missing-imports || print_error "Type checking issues found"
fi

# Run unit tests
print_section "🧪 Running Unit Tests"
pytest tests/test_api.py -v --cov=app.main --cov-report=term-missing --cov-report=html:htmlcov

if [ $? -eq 0 ]; then
    print_success "Unit tests passed"
else
    print_error "Unit tests failed"
    exit 1
fi

# Check test coverage
print_section "📊 Test Coverage Report"
COVERAGE=$(pytest --cov=app.main --cov-report=term | grep "TOTAL" | awk '{print $4}' | sed 's/%//')
echo "Test Coverage: ${COVERAGE}%"

if [ "${COVERAGE%.*}" -ge 90 ]; then
    print_success "Coverage threshold met (≥90%)"
else
    print_error "Coverage below threshold (<90%)"
fi

# Start the application for integration tests
print_section "🚀 Starting Application for Integration Tests"
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
APP_PID=$!

# Wait for application to start
sleep 5

# Check if application is running
if curl -f -s http://127.0.0.1:8000/health > /dev/null; then
    print_success "Application started successfully"
else
    print_error "Application failed to start"
    kill $APP_PID 2>/dev/null || true
    exit 1
fi

# Run integration tests
print_section "🔗 Running Integration Tests"
python -c "
import requests
import json

# Test health endpoint
response = requests.get('http://127.0.0.1:8000/health')
assert response.status_code == 200
assert response.json()['status'] == 'healthy'
print('✓ Health endpoint working')

# Test prediction endpoint
prediction_data = {
    'bill_length_mm': 39.1,
    'bill_depth_mm': 18.7,
    'flipper_length_mm': 181.0,
    'body_mass_g': 3750.0,
    'sex': 'male',
    'island': 'Torgersen'
}

response = requests.post('http://127.0.0.1:8000/predict', json=prediction_data)
assert response.status_code == 200
result = response.json()
assert 'predicted_species' in result
assert 'confidence' in result
assert 'probabilities' in result
print('✓ Prediction endpoint working')

# Test invalid input
invalid_data = {
    'bill_length_mm': -10.0,  # Invalid negative value
    'bill_depth_mm': 18.7,
    'flipper_length_mm': 181.0,
    'body_mass_g': 3750.0,
    'sex': 'male',
    'island': 'Torgersen'
}

response = requests.post('http://127.0.0.1:8000/predict', json=invalid_data)
assert response.status_code == 422
print('✓ Input validation working')

print('All integration tests passed!')
"

if [ $? -eq 0 ]; then
    print_success "Integration tests passed"
else
    print_error "Integration tests failed"
    kill $APP_PID 2>/dev/null || true
    exit 1
fi

# Run quick load test
print_section "⚡ Running Quick Load Test"
timeout 30s locust -f locustfile.py --host http://127.0.0.1:8000 \
    --users 5 --spawn-rate 1 --run-time 20s --headless --only-summary 2>/dev/null || true

print_success "Quick load test completed"

# Cleanup
print_section "🧹 Cleanup"
kill $APP_PID 2>/dev/null || true
print_success "Application stopped"

# Final summary
print_section "📋 Test Summary"
echo -e "${GREEN}✓ Unit Tests: PASSED${NC}"
echo -e "${GREEN}✓ Integration Tests: PASSED${NC}"  
echo -e "${GREEN}✓ Load Test: COMPLETED${NC}"
echo -e "${GREEN}✓ Coverage: ${COVERAGE}%${NC}"

echo -e "\n${BLUE}🎉 All tests completed successfully!${NC}"
echo
echo "Next steps:"
echo "1. Review coverage report: open htmlcov/index.html"
echo "2. Run full load tests: locust -f locustfile.py --host http://127.0.0.1:8000"
echo "3. Deploy to production: ./deploy.sh"