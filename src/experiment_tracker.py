"""
Experiment Tracking Module

Integrates MLflow for tracking model experiments, parameters, and results.

Usage:
    from src.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker(experiment_name="baseline_models")
    with tracker.run(run_name="logistic_regression_v1"):
        tracker.log_params({"learning_rate": 0.01, "max_iter": 2000})
        tracker.log_metrics({"accuracy": 0.85, "f1_score": 0.82})
        tracker.log_artifact(model_path)
"""

import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Optional, Any

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """Track and manage model experiments with MLflow."""
    
    def __init__(self, experiment_name: str = "misinformation_detection", backend_uri: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Parameters
        ----------
        experiment_name : str
            Name of the experiment
        backend_uri : str, optional
            MLflow backend URI (defaults to local)
        """
        self.experiment_name = experiment_name
        self.backend_uri = backend_uri or f"file:{Path.cwd() / 'mlruns'}"
        
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(self.backend_uri)
            try:
                self.experiment = mlflow.get_experiment_by_name(experiment_name)
                if self.experiment:
                    mlflow.set_experiment(experiment_name)
                    print(f"✓ Using existing experiment: {experiment_name}")
                else:
                    mlflow.create_experiment(experiment_name)
                    mlflow.set_experiment(experiment_name)
                    print(f"✓ Created new experiment: {experiment_name}")
            except Exception as e:
                print(f"⚠ MLflow initialization warning: {e}")
        else:
            print("⚠ MLflow not installed. Install with: pip install mlflow")
    
    @contextmanager
    def run(self, run_name: str, description: str = ""):
        """
        Context manager for an MLflow run.
        
        Parameters
        ----------
        run_name : str
            Name of the run
        description : str
            Optional description
        
        Yields
        ------
        None
        """
        if not MLFLOW_AVAILABLE:
            yield
            return
        
        with mlflow.start_run(run_name=run_name):
            if description:
                mlflow.set_tag("description", description)
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            print(f"\n✓ Started MLflow run: {run_name}")
            yield
            print(f"✓ Completed MLflow run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not MLFLOW_AVAILABLE:
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        
        print(f"  • Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not MLFLOW_AVAILABLE:
            return
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        
        print(f"  • Logged {len(metrics)} metrics")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact (model, figure, etc.)."""
        if not MLFLOW_AVAILABLE:
            return
        
        path = Path(artifact_path)
        if path.exists():
            mlflow.log_artifact(artifact_path)
            print(f"  • Logged artifact: {path.name}")
        else:
            print(f"  ⚠ Artifact not found: {artifact_path}")
    
    def log_model(self, model, model_name: str):
        """Log sklearn model."""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            mlflow.sklearn.log_model(model, model_name)
            print(f"  • Logged model: {model_name}")
        except Exception as e:
            print(f"  ⚠ Could not log model: {e}")
    
    def log_dict(self, data: Dict, name: str = "results"):
        """Log dictionary as JSON."""
        if not MLFLOW_AVAILABLE:
            return
        
        # Save to temp file and log
        temp_path = Path(f"/tmp/{name}_{datetime.now().timestamp()}.json")
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        mlflow.log_artifact(str(temp_path))
        temp_path.unlink()  # Clean up
        print(f"  • Logged dict: {name}")
    
    def get_run_info(self) -> Optional[Dict]:
        """Get current run information."""
        if not MLFLOW_AVAILABLE:
            return None
        
        try:
            run = mlflow.active_run()
            if run:
                return {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "start_time": run.info.start_time,
                    "status": run.info.status,
                }
        except Exception as e:
            print(f"⚠ Could not get run info: {e}")
        
        return None


class LocalExperimentTracker:
    """Fallback experiment tracker for local JSON storage when MLflow unavailable."""
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize local experiment tracker.
        
        Parameters
        ----------
        experiment_dir : str
            Directory to store experiments
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_run = None
    
    @contextmanager
    def run(self, run_name: str, description: str = ""):
        """Context manager for an experiment run."""
        self.current_run = {
            "name": run_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
        }
        print(f"\n✓ Started local experiment: {run_name}")
        yield
        
        # Save to file
        run_file = self.experiment_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(run_file, 'w') as f:
            json.dump(self.current_run, f, indent=2)
        
        print(f"✓ Saved experiment: {run_file}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.current_run:
            self.current_run["params"].update(params)
            print(f"  • Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.current_run:
            self.current_run["metrics"].update(metrics)
            print(f"  • Logged {len(metrics)} metrics")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "model"):
        """Log artifact reference."""
        if self.current_run:
            if "artifacts" not in self.current_run:
                self.current_run["artifacts"] = []
            self.current_run["artifacts"].append(str(artifact_path))
            print(f"  • Referenced artifact: {artifact_path}")


def get_tracker(use_mlflow: bool = True, experiment_name: str = "misinformation_detection"):
    """
    Get appropriate experiment tracker (MLflow or Local).
    
    Parameters
    ----------
    use_mlflow : bool
        Whether to use MLflow (falls back to local if unavailable)
    experiment_name : str
        Name of experiment
    
    Returns
    -------
    ExperimentTracker or LocalExperimentTracker
    """
    if use_mlflow and MLFLOW_AVAILABLE:
        return ExperimentTracker(experiment_name)
    else:
        return LocalExperimentTracker()
