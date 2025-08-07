import random
from locust import HttpUser, task, between


class PenguinAPIUser(HttpUser):
    """Simulate users making requests to the Penguin Classification API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Test health endpoint on start
        response = self.client.get("/health")
        if response.status_code != 200:
            self.environment.runner.quit()
    
    @task(10)
    def predict_valid_penguin(self):
        """Test prediction with valid penguin data (most common scenario)"""
        # Generate realistic penguin measurements
        sample_data = self._generate_realistic_penguin_data()
        
        with self.client.post(
            "/predict",
            json=sample_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                result = response.json()
                if "predicted_species" in result and "confidence" in result:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def predict_edge_case_penguin(self):
        """Test prediction with edge case data"""
        edge_data = self._generate_edge_case_penguin_data()
        
        with self.client.post(
            "/predict",
            json=edge_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def test_invalid_data(self):
        """Test API with invalid data (should return 422)"""
        invalid_data = {
            "bill_length_mm": -10.0,  # Invalid negative value
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
            "island": "Torgersen"
        }
        
        with self.client.post(
            "/predict",
            json=invalid_data,
            catch_response=True
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422, got {response.status_code}")
    
    @task(2)
    def test_health_endpoint(self):
        """Test health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service not healthy")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def test_root_endpoint(self):
        """Test root endpoint"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    def _generate_realistic_penguin_data(self):
        """Generate realistic penguin measurement data"""
        # Define realistic ranges for each species
        species_templates = {
            "adelie": {
                "bill_length_mm": (32.1, 46.0),
                "bill_depth_mm": (15.5, 21.5),
                "flipper_length_mm": (172, 210),
                "body_mass_g": (2850, 4775),
                "islands": ["Torgersen", "Biscoe", "Dream"]
            },
            "chinstrap": {
                "bill_length_mm": (40.9, 58.0),
                "bill_depth_mm": (16.4, 20.8),
                "flipper_length_mm": (178, 212),
                "body_mass_g": (2700, 4800),
                "islands": ["Dream"]
            },
            "gentoo": {
                "bill_length_mm": (40.9, 59.6),
                "bill_depth_mm": (13.1, 17.3),
                "flipper_length_mm": (203, 231),
                "body_mass_g": (3950, 6300),
                "islands": ["Biscoe"]
            }
        }
        
        # Randomly select a species template
        species = random.choice(list(species_templates.keys()))
        template = species_templates[species]
        
        return {
            "bill_length_mm": round(random.uniform(*template["bill_length_mm"]), 1),
            "bill_depth_mm": round(random.uniform(*template["bill_depth_mm"]), 1),
            "flipper_length_mm": round(random.uniform(*template["flipper_length_mm"]), 0),
            "body_mass_g": round(random.uniform(*template["body_mass_g"]), 0),
            "sex": random.choice(["male", "female"]),
            "island": random.choice(template["islands"])
        }
    
    def _generate_edge_case_penguin_data(self):
        """Generate edge case penguin data"""
        edge_cases = [
            # Very small penguin
            {
                "bill_length_mm": 25.0,
                "bill_depth_mm": 10.0,
                "flipper_length_mm": 150.0,
                "body_mass_g": 2000.0,
                "sex": "female",
                "island": "Torgersen"
            },
            # Very large penguin
            {
                "bill_length_mm": 70.0,
                "bill_depth_mm": 25.0,
                "flipper_length_mm": 250.0,
                "body_mass_g": 8000.0,
                "sex": "male",
                "island": "Biscoe"
            },
            # Unusual proportions
            {
                "bill_length_mm": 60.0,
                "bill_depth_mm": 12.0,
                "flipper_length_mm": 170.0,
                "body_mass_g": 6000.0,
                "sex": "female",
                "island": "Dream"
            },
            # Minimum valid values
            {
                "bill_length_mm": 0.1,
                "bill_depth_mm": 0.1,
                "flipper_length_mm": 0.1,
                "body_mass_g": 0.1,
                "sex": "male",
                "island": "Torgersen"
            }
        ]
        
        return random.choice(edge_cases)