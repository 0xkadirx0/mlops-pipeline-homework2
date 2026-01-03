"""
MLOps CI/CD Pipeline - Complete Implementation
Homework 2: Implementing the MLOps CI/CD Pipeline

This module contains:
1. Feature Engineering Logic (Unit Test Target)
2. Model Serving Logic (Component Test Target)
3. Smoke Test (Deployment Verification)
4. Unit Tests
5. Component/Integration Tests
6. Smoke Test Implementation

Author: MLOps Student
Date: 2026
"""

import hashlib
import json
import logging
from typing import Dict, List, Tuple, Any
import unittest
from unittest.mock import Mock, patch
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PART 1: FEATURE ENGINEERING LOGIC (Unit Test Target)
# ============================================================================

class HashFeatureEngineer:
    """
    Feature engineering module that implements hashing logic for ML model.
    
    This is the core business logic that will be tested with UNIT TESTS.
    Unit tests are fast and isolated with no external dependencies.
    """
    
    def __init__(self, num_buckets: int = 100):
        """
        Initialize the feature engineer.
        
        Args:
            num_buckets: Number of hash buckets for feature hashing
        """
        self.num_buckets = num_buckets
        logger.info(f"Initialized HashFeatureEngineer with {num_buckets} buckets")
    
    def hash_feature(self, feature_value: str) -> int:
        """
        Hash a feature value to a bucket index.
        
        This is the core feature engineering logic that must be tested.
        
        Args:
            feature_value: String value to hash
            
        Returns:
            Bucket index (0 to num_buckets-1)
            
        Raises:
            ValueError: If feature_value is not a string
        """
        if not isinstance(feature_value, str):
            raise ValueError(f"Feature value must be string, got {type(feature_value)}")
        
        if not feature_value:
            raise ValueError("Feature value cannot be empty")
        
        # Create hash
        hash_object = hashlib.sha256(feature_value.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        
        # Map to bucket
        bucket_index = hash_int % self.num_buckets
        logger.debug(f"Hashed '{feature_value}' to bucket {bucket_index}")
        
        return bucket_index
    
    def extract_features(self, data: Dict[str, Any]) -> Dict[str, int]:
        """
        Extract and hash features from input data.
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Dictionary with hashed feature indices
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        
        features = {}
        for key, value in data.items():
            if isinstance(value, str):
                features[f"{key}_hashed"] = self.hash_feature(value)
            elif isinstance(value, (int, float)):
                features[key] = value
        
        logger.info(f"Extracted {len(features)} features")
        return features
    
    def validate_features(self, features: Dict[str, int]) -> bool:
        """
        Validate that all feature indices are within valid range.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            True if all features are valid
            
        Raises:
            ValueError: If any feature index is out of range
        """
        for key, value in features.items():
            if isinstance(value, int):
                if not (0 <= value < self.num_buckets):
                    raise ValueError(
                        f"Feature {key} has invalid bucket index {value}. "
                        f"Must be between 0 and {self.num_buckets-1}"
                    )
        
        logger.info("All features validated successfully")
        return True


# ============================================================================
# PART 2: MODEL SERVING LOGIC (Component Test Target)
# ============================================================================

class MockCardinalityPredictor:
    """
    Mock High-Cardinality Prediction Service.
    
    This simulates the model serving logic that will be tested with
    COMPONENT/INTEGRATION TESTS. These tests verify interaction between
    model serving logic and data sources.
    """
    
    def __init__(self, model_weights: Dict[str, float] = None):
        """
        Initialize the predictor with model weights.
        
        Args:
            model_weights: Pre-trained model weights
        """
        self.model_weights = model_weights or {
            "feature_0": 0.5,
            "feature_1": 0.3,
            "feature_2": 0.2
        }
        self.feature_engineer = HashFeatureEngineer(num_buckets=100)
        logger.info("Initialized MockCardinalityPredictor")
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction based on input data.
        
        This method represents the model serving logic that depends on:
        - Feature engineering (internal)
        - Data consistency (external)
        
        Args:
            input_data: Input features for prediction
            
        Returns:
            Prediction result with confidence
        """
        try:
            # Extract features
            features = self.feature_engineer.extract_features(input_data)
            
            # Validate features
            self.feature_engineer.validate_features(features)
            
            # Make prediction (simplified mock)
            prediction_score = sum(
                self.model_weights.get(f"feature_{i}", 0.1)
                for i in range(len(features))
            ) / len(features) if features else 0.0
            
            result = {
                "status": "success",
                "prediction": prediction_score,
                "confidence": min(0.95, prediction_score * 1.1),
                "features_count": len(features)
            }
            
            logger.info(f"Prediction made: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "prediction": None
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint for deployment verification.
        
        Returns:
            Health status
        """
        return {
            "status": "healthy",
            "service": "CardinalityPredictor",
            "version": "1.0.0"
        }


# ============================================================================
# PART 3: UNIT TESTS (Fast, Isolated, No External Dependencies)
# ============================================================================

class TestHashFeatureEngineer(unittest.TestCase):
    """
    Unit tests for HashFeatureEngineer.
    
    These tests are FAST because they:
    - Have no external dependencies (no database, no network)
    - Test isolated logic (pure functions)
    - Run in memory
    - Complete in milliseconds
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = HashFeatureEngineer(num_buckets=100)
    
    def test_hash_feature_returns_valid_bucket_index(self):
        """
        Test that hash_feature returns a valid bucket index.
        
        This is the core test mentioned in the homework:
        "ensure your HashFeature function returns the correct bucket index
        for a known input string"
        """
        # Known input string
        test_input = "user_123"
        
        # Get bucket index
        bucket_index = self.engineer.hash_feature(test_input)
        
        # Assertions
        self.assertIsInstance(bucket_index, int)
        self.assertGreaterEqual(bucket_index, 0)
        self.assertLess(bucket_index, 100)
        logger.info(f"✓ Test passed: hash_feature('{test_input}') = {bucket_index}")
    
    def test_hash_feature_consistency(self):
        """Test that the same input always produces the same output."""
        test_input = "consistent_value"
        
        result1 = self.engineer.hash_feature(test_input)
        result2 = self.engineer.hash_feature(test_input)
        
        self.assertEqual(result1, result2)
        logger.info(f"✓ Test passed: hash consistency verified")
    
    def test_hash_feature_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs (usually)."""
        result1 = self.engineer.hash_feature("input_1")
        result2 = self.engineer.hash_feature("input_2")
        
        # Note: collision is theoretically possible but unlikely
        self.assertNotEqual(result1, result2)
        logger.info(f"✓ Test passed: different inputs produce different outputs")
    
    def test_hash_feature_invalid_input_type(self):
        """Test that non-string inputs raise ValueError."""
        with self.assertRaises(ValueError):
            self.engineer.hash_feature(123)
        
        with self.assertRaises(ValueError):
            self.engineer.hash_feature(None)
        
        logger.info(f"✓ Test passed: invalid input types rejected")
    
    def test_hash_feature_empty_string(self):
        """Test that empty strings raise ValueError."""
        with self.assertRaises(ValueError):
            self.engineer.hash_feature("")
        
        logger.info(f"✓ Test passed: empty strings rejected")
    
    def test_extract_features(self):
        """Test feature extraction from dictionary."""
        data = {
            "user_id": "user_123",
            "category": "premium",
            "age": 25
        }
        
        features = self.engineer.extract_features(data)
        
        self.assertIn("user_id_hashed", features)
        self.assertIn("category_hashed", features)
        self.assertIn("age", features)
        self.assertEqual(features["age"], 25)
        logger.info(f"✓ Test passed: feature extraction successful")
    
    def test_validate_features_valid(self):
        """Test validation of valid features."""
        valid_features = {
            "feature_0": 50,
            "feature_1": 99,
            "feature_2": 0
        }
        
        result = self.engineer.validate_features(valid_features)
        self.assertTrue(result)
        logger.info(f"✓ Test passed: valid features accepted")
    
    def test_validate_features_invalid(self):
        """Test validation rejects out-of-range features."""
        invalid_features = {
            "feature_0": 100  # Out of range (should be 0-99)
        }
        
        with self.assertRaises(ValueError):
            self.engineer.validate_features(invalid_features)
        
        logger.info(f"✓ Test passed: invalid features rejected")


# ============================================================================
# PART 4: COMPONENT/INTEGRATION TESTS (With External Dependencies)
# ============================================================================

class TestModelServingLogic(unittest.TestCase):
    """
    Component/Integration tests for model serving logic.
    
    These tests are END-TO-END because they:
    - Test interaction between model serving and data sources
    - Can involve database or file system
    - Verify data consistency across components
    - Take longer than unit tests
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = MockCardinalityPredictor()
    
    def test_predict_with_valid_input(self):
        """
        Test prediction with valid input data.
        
        This verifies the interaction between:
        - Feature engineering logic
        - Model serving logic
        - Data consistency
        """
        input_data = {
            "user_id": "user_456",
            "category": "standard",
            "score": 75
        }
        
        result = self.predictor.predict(input_data)
        
        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["prediction"])
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 1)
        logger.info(f"✓ Component test passed: prediction successful")
    
    def test_predict_with_invalid_input(self):
        """Test prediction handles invalid input gracefully."""
        # This tests error handling in the serving logic
        result = self.predictor.predict(None)
        
        self.assertEqual(result["status"], "error")
        logger.info(f"✓ Component test passed: error handling verified")
    
    def test_health_check(self):
        """Test health check endpoint."""
        health = self.predictor.health_check()
        
        self.assertEqual(health["status"], "healthy")
        self.assertIn("service", health)
        self.assertIn("version", health)
        logger.info(f"✓ Component test passed: health check successful")
    
    def test_data_consistency(self):
        """
        Test data consistency across multiple predictions.
        
        This verifies that the model produces consistent results
        for the same input across multiple calls.
        """
        input_data = {"user_id": "consistent_user", "value": 100}
        
        result1 = self.predictor.predict(input_data)
        result2 = self.predictor.predict(input_data)
        
        self.assertEqual(result1["prediction"], result2["prediction"])
        logger.info(f"✓ Component test passed: data consistency verified")


# ============================================================================
# PART 5: SMOKE TEST (Deployment Verification)
# ============================================================================

class SmokeTest:
    """
    Smoke Test for deployment verification.
    
    This is the critical "Deployment Test" mentioned in the homework.
    It verifies that:
    - The service starts up successfully
    - The service responds to requests
    - The service returns expected status codes
    
    This is END-TO-END because it tests the entire deployed system.
    """
    
    @staticmethod
    def test_service_health() -> bool:
        """
        Test that the service is healthy and responsive.
        
        This simulates: curl -X GET http://localhost:5000/health
        Or: python script that sends a prediction request
        
        Returns:
            True if service is healthy (HTTP 200)
        """
        try:
            predictor = MockCardinalityPredictor()
            health = predictor.health_check()
            
            # Verify response
            assert health["status"] == "healthy", "Service not healthy"
            assert "service" in health, "Missing service info"
            
            logger.info("✓ Smoke test passed: Service is healthy (200 OK)")
            return True
            
        except Exception as e:
            logger.error(f"✗ Smoke test failed: {str(e)}")
            return False
    
    @staticmethod
    def test_service_prediction() -> bool:
        """
        Test that the service can make predictions.
        
        This simulates: curl -X POST http://localhost:5000/predict -d '...'
        
        Returns:
            True if prediction succeeds (HTTP 200)
        """
        try:
            predictor = MockCardinalityPredictor()
            
            test_input = {
                "user_id": "smoke_test_user",
                "category": "test",
                "value": 42
            }
            
            result = predictor.predict(test_input)
            
            # Verify response
            assert result["status"] == "success", "Prediction failed"
            assert result["prediction"] is not None, "No prediction returned"
            
            logger.info("✓ Smoke test passed: Service returns 200 OK with valid prediction")
            return True
            
        except Exception as e:
            logger.error(f"✗ Smoke test failed: {str(e)}")
            return False


# ============================================================================
# PART 6: INTENTIONAL BUG FOR "STOP THE LINE" SIMULATION
# ============================================================================

class BuggyHashFeatureEngineer:
    """
    INTENTIONAL BUG VERSION - For demonstrating CI/CD pipeline failure.
    
    This version contains a syntax error or logic bug that will cause
    the CI pipeline to fail, demonstrating the "Stop the Line" principle.
    
    Uncomment the bug below to see the pipeline fail.
    """
    
    def __init__(self, num_buckets: int = 100):
        self.num_buckets = num_buckets
    
    def hash_feature_with_bug(self, feature_value: str) -> int:
        """
        BUGGY VERSION: Contains intentional error.
        
        The bug: Missing validation that causes TypeError
        """
        # BUG: This will fail if feature_value is not a string
        # because we don't validate the input type
        hash_object = hashlib.sha256(feature_value.encode())  # Will crash if feature_value is None
        hash_int = int(hash_object.hexdigest(), 16)
        return hash_int % self.num_buckets
    
    def syntax_error_example(self):
        """
        SYNTAX ERROR EXAMPLE - Uncomment to trigger linting failure.
        
        This demonstrates how code analysis tools catch syntax errors.
        """
        # x = 5  # Missing closing parenthesis would be caught by linter
        # if x > 3
        #     print("This is a syntax error")  # Missing colon
        pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_tests():
    """Run all unit and component tests."""
    logger.info("=" * 70)
    logger.info("STARTING MLOps CI/CD PIPELINE TEST SUITE")
    logger.info("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all tests
    suite.addTests(loader.loadTestsFromTestCase(TestHashFeatureEngineer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelServingLogic))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_smoke_tests():
    """Run smoke tests for deployment verification."""
    logger.info("=" * 70)
    logger.info("STARTING SMOKE TESTS (DEPLOYMENT VERIFICATION)")
    logger.info("=" * 70)
    
    smoke = SmokeTest()
    
    health_ok = smoke.test_service_health()
    prediction_ok = smoke.test_service_prediction()
    
    logger.info("=" * 70)
    if health_ok and prediction_ok:
        logger.info("✓ ALL SMOKE TESTS PASSED - Service is ready for production")
        return True
    else:
        logger.error("✗ SMOKE TESTS FAILED - Deployment blocked")
        return False


if __name__ == "__main__":
    # Run all tests
    unit_tests_passed = run_all_tests()
    
    logger.info("\n")
    
    # Run smoke tests
    smoke_tests_passed = run_smoke_tests()
    
    # Exit with appropriate code
    if unit_tests_passed and smoke_tests_passed:
        logger.info("\n✓ ALL TESTS PASSED - Pipeline successful")
        sys.exit(0)
    else:
        logger.error("\n✗ TESTS FAILED - Pipeline blocked")
        sys.exit(1)
