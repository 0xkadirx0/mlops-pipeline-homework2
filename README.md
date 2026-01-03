# MLOps CI/CD Pipeline - Homework 2 Implementation

**Deadline:** 11.01.2026 Sunday 23:59

## 📋 Overview

This project implements a complete **MLOps CI/CD Pipeline** that transitions from manual ML workflows (MLOps Level 0) to fully automated pipelines (MLOps Level 1 & 2). The implementation includes:

- **Commit Stage (Continuous Integration):** Automated code quality checks
- **Acceptance Gate (Continuous Deployment):** Build, package, and smoke testing
- **Stop the Line Simulation:** Demonstrates pipeline failure blocking deployment

---

## 🎯 Project Structure

```
mlops-pipeline/
├── mlops_pipeline.py          # Complete implementation (all-in-one file)
├── .github/
│   └── workflows/
│       └── main.yml           # GitHub Actions CI/CD workflow
├── README.md                  # This file
├── requirements.txt           # Python dependencies
└── Dockerfile                 # Container configuration
```

---

## 📦 Components

### 1. **Feature Engineering Logic** (Unit Test Target)
```python
class HashFeatureEngineer:
    """Core business logic for ML feature engineering."""
    - hash_feature(feature_value): Hash string to bucket index
    - extract_features(data): Extract and hash features from input
    - validate_features(features): Validate feature indices
```

**Why it's tested with UNIT TESTS:**
- Fast execution (milliseconds)
- Isolated logic with no external dependencies
- Pure functions with deterministic outputs
- No database or network calls

**Example Test:**
```python
def test_hash_feature_returns_valid_bucket_index(self):
    """Ensure HashFeature returns correct bucket index for known input."""
    bucket_index = self.engineer.hash_feature("user_123")
    assert 0 <= bucket_index < 100
```

---

### 2. **Model Serving Logic** (Component Test Target)
```python
class MockCardinalityPredictor:
    """ML model serving logic."""
    - predict(input_data): Make predictions using model
    - health_check(): Verify service health
```

**Why it's tested with COMPONENT/INTEGRATION TESTS:**
- Tests interaction between model serving and data sources
- Can involve database or file system
- Verifies data consistency across components
- Takes longer than unit tests (seconds)

**Example Test:**
```python
def test_predict_with_valid_input(self):
    """Test prediction with valid input data."""
    result = self.predictor.predict({"user_id": "user_456", ...})
    assert result["status"] == "success"
    assert result["prediction"] is not None
```

---

### 3. **Smoke Test** (Deployment Verification)
```python
class SmokeTest:
    """Critical deployment verification tests."""
    - test_service_health(): Verify service responds (HTTP 200)
    - test_service_prediction(): Verify prediction endpoint works
```

**Why it's an END-TO-END test:**
- Tests the entire deployed system
- Simulates real user requests
- Verifies service availability
- Critical for production deployment

**Example:**
```python
def test_service_health():
    """Simulate: curl -X GET http://localhost:5000/health"""
    health = predictor.health_check()
    assert health["status"] == "healthy"  # Returns 200 OK
```

---

## 🔄 CI/CD Pipeline Stages

### **Stage 1: Commit Stage (Continuous Integration)**

Runs on every commit to ensure code quality:

```
┌─────────────────────────────────────────┐
│  1. Unit Tests (Fast, Isolated)         │
│     - HashFeature logic tests           │
│     - No external dependencies          │
│     - Execution time: ~100ms            │
├─────────────────────────────────────────┤
│  2. Component Tests (Integration)       │
│     - Model serving logic tests         │
│     - Data consistency verification     │
│     - Execution time: ~500ms            │
├─────────────────────────────────────────┤
│  3. Code Linting (Pylint)               │
│     - Check code style                  │
│     - Detect syntax errors              │
│     - Enforce quality thresholds        │
├─────────────────────────────────────────┤
│  4. Code Analysis (Flake8)              │
│     - Additional style checks           │
│     - Complexity analysis               │
├─────────────────────────────────────────┤
│  5. Code Formatting (Black)             │
│     - Verify consistent formatting      │
├─────────────────────────────────────────┤
│  6. Test Coverage Analysis              │
│     - Measure code coverage             │
│     - Generate coverage reports         │
└─────────────────────────────────────────┘
         ↓ (All pass)
    ✓ PROCEED TO NEXT STAGE
         ↓ (Any fail)
    ✗ STOP THE LINE - Block deployment
```

### **Stage 2: Acceptance Gate (Continuous Deployment)**

Only runs if Commit Stage passes:

```
┌─────────────────────────────────────────┐
│  1. Build & Package                     │
│     - Create deployable artifacts       │
│     - "Only build your binaries once"   │
├─────────────────────────────────────────┤
│  2. Docker Build                        │
│     - Containerize application          │
│     - Create deployment image           │
├─────────────────────────────────────────┤
│  3. Smoke Test - Health Check           │
│     - Verify service starts (HTTP 200)  │
│     - Critical deployment test          │
├─────────────────────────────────────────┤
│  4. Smoke Test - Prediction             │
│     - Send test prediction request      │
│     - Verify response (HTTP 200)        │
└─────────────────────────────────────────┘
         ↓ (All pass)
    ✓ DEPLOYMENT APPROVED
         ↓ (Any fail)
    ✗ DEPLOYMENT BLOCKED
```

### **Stage 3: Stop the Line Simulation**

Demonstrates CI/CD pipeline failure:

```
Developer commits code with intentional bug
           ↓
CI Pipeline detects failure
           ↓
✗ Pipeline STOPS
           ↓
✗ Deployment is BLOCKED
           ↓
Developer must fix bug before deployment
           ↓
Re-commit fixed code
           ↓
Pipeline runs again
           ↓
✓ All tests pass
           ↓
✓ Deployment proceeds
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.11+
python --version

# Install dependencies
pip install pytest pylint flake8 black
```

### Run All Tests Locally
```bash
# Run all unit and component tests + smoke tests
python mlops_pipeline.py

# Expected output:
# ✓ 12 tests passed
# ✓ ALL SMOKE TESTS PASSED
# ✓ ALL TESTS PASSED - Pipeline successful
```

### Run Specific Test Suites
```bash
# Unit tests only
pytest mlops_pipeline.py::TestHashFeatureEngineer -v

# Component tests only
pytest mlops_pipeline.py::TestModelServingLogic -v

# With coverage
pytest mlops_pipeline.py -v --cov=. --cov-report=term-missing
```

### Code Quality Checks
```bash
# Pylint analysis
pylint mlops_pipeline.py

# Flake8 style check
flake8 mlops_pipeline.py

# Black formatting check
black --check mlops_pipeline.py
```

---

## 📊 Test Results

### ✓ Success Case (All Tests Pass)

```
======================================================================
STARTING MLOps CI/CD PIPELINE TEST SUITE
======================================================================

test_hash_feature_returns_valid_bucket_index ... ok
test_hash_feature_consistency ... ok
test_hash_feature_different_inputs_different_outputs ... ok
test_hash_feature_invalid_input_type ... ok
test_hash_feature_empty_string ... ok
test_extract_features ... ok
test_validate_features_valid ... ok
test_validate_features_invalid ... ok
test_predict_with_valid_input ... ok
test_predict_with_invalid_input ... ok
test_health_check ... ok
test_data_consistency ... ok

Ran 12 tests in 0.002s

======================================================================
STARTING SMOKE TESTS (DEPLOYMENT VERIFICATION)
======================================================================

✓ Smoke test passed: Service is healthy (200 OK)
✓ Smoke test passed: Service returns 200 OK with valid prediction

======================================================================
✓ ALL SMOKE TESTS PASSED - Service is ready for production
✓ ALL TESTS PASSED - Pipeline successful
```

### ✗ Failure Case (Stop the Line)

When a bug is introduced:

```
FAILED mlops_pipeline.py::TestHashFeatureEngineer::test_hash_feature_invalid_input_type

ValueError: Feature value must be string, got <class 'int'>

======================================================================
✗ TESTS FAILED - Pipeline blocked
Deployment is prevented until bug is fixed
```

---

## 🐳 Docker Integration

### Build Docker Image
```bash
docker build -t mlops-pipeline:latest .
```

### Run Container
```bash
docker run -it mlops-pipeline:latest python mlops_pipeline.py
```

### Run Smoke Test Against Container
```bash
# Start container in background
docker run -d --name mlops-service mlops-pipeline:latest

# Run smoke tests
docker exec mlops-service python -c "
from mlops_pipeline import SmokeTest
SmokeTest.test_service_health()
SmokeTest.test_service_prediction()
"
```

---

## 📝 Code Examples

### Unit Test Example (Fast, Isolated)
```python
def test_hash_feature_returns_valid_bucket_index(self):
    """
    Test that hash_feature returns correct bucket index.
    
    Why this is FAST:
    - No database calls
    - No network requests
    - Pure function with deterministic output
    - Execution time: ~1ms
    """
    test_input = "user_123"
    bucket_index = self.engineer.hash_feature(test_input)
    
    assert isinstance(bucket_index, int)
    assert 0 <= bucket_index < 100
```

### Component Test Example (End-to-End)
```python
def test_predict_with_valid_input(self):
    """
    Test prediction with valid input data.
    
    Why this is END-TO-END:
    - Tests interaction between components
    - Verifies data consistency
    - Simulates real usage scenario
    - Execution time: ~50ms
    """
    input_data = {
        "user_id": "user_456",
        "category": "standard",
        "score": 75
    }
    
    result = self.predictor.predict(input_data)
    
    assert result["status"] == "success"
    assert result["prediction"] is not None
    assert 0 <= result["confidence"] <= 1
```

### Smoke Test Example (Deployment Verification)
```python
def test_service_health():
    """
    Smoke test for deployment verification.
    
    Simulates: curl -X GET http://localhost:5000/health
    
    Why this is CRITICAL:
    - Verifies service is running
    - Confirms HTTP 200 response
    - Blocks deployment if fails
    """
    predictor = MockCardinalityPredictor()
    health = predictor.health_check()
    
    assert health["status"] == "healthy"
    assert "service" in health
    
    return True  # HTTP 200 OK
```

---

## 🛑 Stop the Line Demonstration

### Intentional Bug Introduction
```python
# Commit code with intentional bug
class BuggyHashFeatureEngineer:
    def hash_feature_with_bug(self, feature_value: str) -> int:
        # BUG: Missing validation
        hash_object = hashlib.sha256(feature_value.encode())
        # Will crash if feature_value is None
        return hash_int % self.num_buckets
```

### Pipeline Detection
```
✗ TESTS FAILED
✗ Pipeline blocked deployment
✗ Bad code prevented from entering production
```

### Developer Fix
```python
# Fix the bug
def hash_feature(self, feature_value: str) -> int:
    # Proper validation
    if not isinstance(feature_value, str):
        raise ValueError(f"Feature value must be string")
    
    hash_object = hashlib.sha256(feature_value.encode())
    return hash_int % self.num_buckets
```

### Pipeline Re-run
```
✓ All tests pass
✓ Pipeline proceeds to deployment
✓ Fixed code deployed to production
```

---

## 📋 Deliverables Checklist

- [x] **Pipeline Configuration**
  - GitHub Actions workflow file (.github/workflows/main.yml)
  - Shows all stages: Build → Unit Test → Lint → Package → Smoke Test

- [x] **Test Results**
  - Evidence A (Success): Green build with all tests passing
  - Evidence B (Failure): Pipeline blocking deployment after bug introduction

- [x] **Test Code**
  - Unit Test code: `TestHashFeatureEngineer` class
  - Smoke Test code: `SmokeTest` class
  - Component Test code: `TestModelServingLogic` class
  - Explanations of why each test type is used

- [x] **Complete Implementation**
  - All code in single Python file (mlops_pipeline.py)
  - GitHub Actions workflow configuration
  - Docker support
  - Comprehensive documentation

---

## 🔑 Key Concepts

### Unit Tests (Fast)
- **Purpose:** Test isolated business logic
- **Characteristics:** Fast, deterministic, no external dependencies
- **Example:** HashFeature function tests
- **Execution Time:** Milliseconds
- **CI Stage:** Commit Stage

### Component/Integration Tests (End-to-End)
- **Purpose:** Test interaction between components
- **Characteristics:** Can involve database/file system, verify data consistency
- **Example:** Model serving with data source interaction
- **Execution Time:** Seconds
- **CI Stage:** Commit Stage

### Smoke Tests (Deployment Verification)
- **Purpose:** Verify deployed system works
- **Characteristics:** Tests entire system, simulates real user requests
- **Example:** Health check, prediction endpoint
- **Execution Time:** Seconds
- **CI Stage:** Acceptance Gate

### Stop the Line Principle
- **Core Practice:** If any pipeline stage fails, deployment is blocked
- **Benefit:** Bad code is prevented from entering production
- **Implementation:** Pipeline exits with error code on failure
- **Result:** Developers must fix issues before deployment

---

## 📚 References

- [MLOps Principles](https://ml-ops.systems/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)
- [CI/CD Best Practices](https://www.atlassian.com/continuous-delivery/principles/continuous-integration-vs-delivery-vs-deployment)

---

## 👤 Author

MLOps Student
Date: January 2026

---

## 📄 License

This project is provided for educational purposes as part of the MLOps course.

---

## ❓ Questions?

Refer to the inline code documentation in `mlops_pipeline.py` for detailed explanations of each component.
