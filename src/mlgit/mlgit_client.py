import datetime
import json
import shutil
import os

import pandas as pd

from pyutils.pickable import PickableObject
from pyutils.git import *

class MLGitClient:
    """
    Repository Architecture
        - registry_dpath
            - model_name
                - model_artifacts *
                - backtest
                - model_versions *
                    - model
                    - model_version_artifacts *
    """
    def __init__(self, user_name: str, repo_name: str, registry_dpath: str = None):
        self.user_name = user_name
        self.repo_name = repo_name
        self.registry_dpath = registry_dpath

    def model_remote_path(self, model_name: str, model_version: str = None,
        artifact_name: str = None) -> str:
        return '/'.join([
            path_component for path_component in
            [self.registry_dpath, model_name, model_version, artifact_name]
            if path_component is not None
        ])

    def get_version_list(self, model_name: str) -> list:
        return self.get_json_artifact("versions", model_name)

    def get_json_artifact(self, artifact_name: str, model_name: str,
        model_version: any = None) -> any:
        remote_artifact_fpath = self.model_remote_path(model_name, model_version,
                f"{artifact_name}.json")

        return json.loads(
            read_remote_file(self.user_name, self.repo_name, remote_artifact_fpath)
        )

    def get_pandas_artifact(self, artifact_name: str, model_name: str,
        model_version: str = None, **read_csv_kwargs) -> pd.DataFrame:
        remote_artifact_fpath = self.model_remote_path(model_name, model_version,
                f"{artifact_name}.csv")
        
        return pd.read_csv(
            web_remote_fpath(self.user_name, self.repo_name, remote_artifact_fpath),
            **read_csv_kwargs
        )

    def get_model_backtest(self, model_name: str) -> pd.DataFrame:
        model_backtest = self.get_pandas_artifact("backtest", model_name)
        
        index_name = model_backtest.columns[0]
        model_backtest[index_name] = pd.to_datetime(model_backtest[index_name])
        model_backtest["version_timestamp"] = pd.to_datetime(model_backtest["version_timestamp"])
        model_backtest = model_backtest.set_index(index_name)

        return model_backtest

    def get_model_version(self, model_name: str, model_version: str) -> any:
        model_version_local_dpath = os.mkdir(os.path.join(os.getcwd(), "temp_model_version"))

        pull_directory(
            self.user_name, self.repo_name,
            self.model_remote_path(model_name, model_version),
            model_version_local_dpath
        )

        pickable_model = PickableObject.restore(os.path.join(model_version_local_dpath, "model"))
        shutil.rmtree(model_version_local_dpath)
        return pickable_model

    # Artifact Logging operations
    def register_model(self, access_token: str, model_name: str) -> None:
        self.log_json_artifact(
            access_token=access_token,
            json_artifact=[],
            artifact_name="versions",
            model_name=model_name
        )

    def log_artifact(self, access_token: str, artifact_fpath: str,
        model_name: str, model_version: str = None) -> None:
        push_files(
            access_token=access_token,
            repo_name=self.repo_name,
            from_local_fpaths=[artifact_fpath],
            to_remote_dpaths=self.model_remote_path(model_name, model_version)
        )

    def log_json_artifact(self, access_token: str, json_artifact: any,
        artifact_name: str, model_name: str, model_version: str = None) -> None:
        
        artifact_temp_fpath = os.path.join(os.getcwd(), f"{artifact_name}.json")

        with open(artifact_temp_fpath, 'w') as artifact_file:
            json.dump(json_artifact, artifact_file)

        self.log_artifact(access_token, artifact_temp_fpath, model_name, model_version)
        os.remove(artifact_temp_fpath)

    def log_pandas_artifact(self, access_token: str, pandas_artifact: pd.DataFrame,
        artifact_name: str, model_name: str, model_version: str = None,
        **to_csv_kwargs) -> any:
        
        artifact_temp_fpath = os.path.join(os.getcwd(), f"{artifact_name}.csv")
        pandas_artifact.to_csv(artifact_temp_fpath, **to_csv_kwargs)

        self.log_artifact(access_token, artifact_temp_fpath, model_name, model_version)
        os.remove(artifact_temp_fpath)

    def log_model_backtest(self, access_token: str, model_backtest: pd.DataFrame,
        model_name: str, version_timestamp: datetime.datetime = None) -> None:
        """ 
        """
        if version_timestamp is None: # Set to most recent prediction
            version_timestamp = model_backtest.index.max()

        # Ensure compatibility of datetime comparison operations
        if isinstance(model_backtest.index, pd.PeriodIndex):
            model_backtest.index = model_backtest.index.to_timestamp()

        if isinstance(version_timestamp, pd.Period):
            version_timestamp = version_timestamp.to_timestamp()

        if model_backtest.index.name is None:
            model_backtest.index.name = "date"

        model_backtest.columns = model_backtest.columns.astype(str)
        model_backtest["version_timestamp"] = version_timestamp
        
        try:
            prior_model_backtest = self.get_model_backtest(model_name)
        except:
            return self.log_pandas_artifact(access_token, model_backtest,
                    "backtest", model_name)

        # Override values
        prior_model_backtest[
            (prior_model_backtest.index.isin(model_backtest.index)) &
            (prior_model_backtest["version_timestamp"] > version_timestamp) &
            (prior_model_backtest.index < prior_model_backtest["version_timestamp"])
        ] = model_backtest

        new_model_backtest = pd.concat([
            prior_model_backtest,
            model_backtest[~model_backtest.index.isin(prior_model_backtest.index)]
        ], axis=0).sort_index()
        
        self.log_pandas_artifact(access_token, new_model_backtest, "backtest", model_name)

    # Version Logging operations
    def log_model_version(self, access_token: str, pickable_model: PickableObject,
        model_name: str, model_version: str) -> None:

        model_version_local_dpath, model_version_local_fpath = \
                self.make_model_version_local_paths(model_version)

        pickable_model.save(model_version_local_fpath)
        self.log_model_version_from_local(access_token, model_name, model_version,
                model_version_local_dpath)
        
        shutil.rmtree(model_version_local_dpath)

    def make_model_version_local_paths(self, model_version: str) -> tuple:
        model_version_local_dpath = os.path.join(os.getcwd(), model_version)

        if not os.path.exists(model_version_local_dpath):
            os.makedirs(model_version_local_dpath)

        return model_version_local_dpath, \
                os.path.join(model_version_local_dpath, "model")

    def log_model_version_from_local(self, access_token: str, model_name: str,
        model_version: str, model_version_local_dpath: str) -> None:
        push_directory(
            access_token=access_token,
            repo_name=self.repo_name,
            from_local_dpath=model_version_local_dpath,
            to_remote_dpath=self.model_remote_path(model_name, model_version),
            timeout=120
        )

        model_versions = self.get_version_list(model_name)
        model_versions.append(model_version)
        self.log_json_artifact(access_token, model_versions, "versions", model_name)

if __name__ == "__main__":
    pass
