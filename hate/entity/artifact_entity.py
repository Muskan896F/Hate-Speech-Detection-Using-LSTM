from dataclasses import dataclass

# ✅ Data ingestion artifact
@dataclass
class DataIngestionArtifacts:
    raw_data_file_path: str
    imbalance_data_file_path: str

# ✅ Data transformation artifact
@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str

# ✅ Model trainer artifact
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    x_test_path: str
    y_test_path: str

# ✅ Model evaluation artifact
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool

# ✅ Model pusher artifact
@dataclass
class ModelPusherArtifacts:
    pushed_model_dir: str
    model_file_path: str
