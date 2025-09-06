from src.CancerClassifier.constants.constant import *
from src.CancerClassifier.utils.common import read_yaml, create_directories , save_json
import tensorflow as tf
import mlflow
import mlflow.keras
import dagshub
from pathlib import Path
from src.CancerClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)  
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        dagshub.init(
            repo_owner=self.config.dagshub_repo_owner,
            repo_name=self.config.dagshub_repo_name,
            mlflow=True,
        )
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": float(self.score[0]), "accuracy": float(self.score[1])})
            # Model registry does not work with file store
            mlflow.keras.log_model(self.model, artifact_path="model", registered_model_name="VGG16Model")

