from prometheus_client import Counter, Histogram, Gauge

# =========================================================
# 🔹 API METRICS (FastAPI)
# =========================================================
API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status"]
)

API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"]
)

API_REQUESTS_IN_PROGRESS = Gauge(
    "api_requests_in_progress",
    "Number of requests in progress"
)

API_ERRORS_TOTAL = Counter(
    "api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"]
)

# =========================================================
# 🔹 AUTH / SECURITY METRICS
# =========================================================
AUTH_LOGIN_TOTAL = Counter(
    "auth_login_total",
    "Total successful logins"
)

AUTH_FAILED_TOTAL = Counter(
    "auth_failed_total",
    "Total failed login attempts"
)

AUTH_TOKEN_EXPIRED = Counter(
    "auth_token_expired_total",
    "Expired JWT tokens"
)

SECURITY_SUSPICIOUS_REQUESTS = Counter(
    "security_suspicious_requests_total",
    "Suspicious requests detected"
)

# =========================================================
# 🔹 UPLOAD METRICS
# =========================================================
IMAGE_UPLOAD_TOTAL = Counter(
    "image_upload_total",
    "Total uploaded images"
)

IMAGE_UPLOAD_SIZE = Histogram(
    "image_upload_size_bytes",
    "Uploaded image sizes"
)

# =========================================================
# 🔹 YOLO DETECTION METRICS
# =========================================================
YOLO_DETECTIONS_TOTAL = Counter(
    "yolo_detections_total",
    "Total YOLO detections",
    ["class_name"]
)

YOLO_OBJECTS_DETECTED = Counter(
    "yolo_objects_detected",
    "Number of detected objects",
    ["class_name"]
)

YOLO_INFERENCE_TIME = Histogram(
    "yolo_inference_time_seconds",
    "YOLO inference time"
)

YOLO_CONFIDENCE_SCORE = Gauge(
    "yolo_confidence_score",
    "YOLO confidence score",
    ["class_name"]
)

# =========================================================
# 🔹 CLASSIFICATION METRICS (MATURITY / VARIETY)
# =========================================================
CLASSIFICATION_PREDICTIONS_TOTAL = Counter(
    "classification_predictions_total",
    "Total classification predictions",
    ["type"]  # maturity / variety
)

CLASSIFICATION_CONFIDENCE = Gauge(
    "classification_confidence",
    "Classification confidence score",
    ["type"]
)

CLASSIFICATION_ERRORS = Counter(
    "classification_errors_total",
    "Classification errors",
    ["type"]
)

# =========================================================
# 🔹 LLM / RAG METRICS
# =========================================================
LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "Total LLM requests"
)

LLM_LATENCY = Histogram(
    "llm_latency_seconds",
    "LLM response latency"
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "LLM errors"
)

LLM_RESPONSE_LENGTH = Histogram(
    "llm_response_length",
    "Length of LLM responses"
)

LLM_QUALITY_SCORE = Gauge(
    "llm_quality_score",
    "Quality score of LLM responses"
)

# =========================================================
# 🔹 AIRFLOW METRICS
# =========================================================
AIRFLOW_DAG_RUNS = Counter(
    "airflow_dag_runs_total",
    "Total DAG runs",
    ["dag_id"]
)

AIRFLOW_DAG_SUCCESS = Counter(
    "airflow_dag_success_total",
    "Successful DAG runs",
    ["dag_id"]
)

AIRFLOW_DAG_FAILED = Counter(
    "airflow_dag_failed_total",
    "Failed DAG runs",
    ["dag_id"]
)

AIRFLOW_TASK_DURATION = Histogram(
    "airflow_task_duration_seconds",
    "Task duration",
    ["task_id"]
)

AIRFLOW_DAG_DURATION = Gauge(
    "airflow_dag_duration_seconds",
    "DAG execution duration",
    ["dag_id"]
)

# =========================================================
# 🔹 BUSINESS METRICS
# =========================================================
DATES_DETECTED_TOTAL = Counter(
    "dates_detected_total",
    "Total detected dates",
    ["variety", "maturity"]
)

PREDICTIONS_PER_USER = Counter(
    "predictions_per_user_total",
    "Predictions per user",
    ["user_id"]
)