import orjson
from numbers import Number
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from numpy import isnan

from matchescu.adt.entity_resolution_result import EntityResolutionResult
from matchescu.cli._generate import generate
from matchescu.common.partitioning import compute_partition


class ExperimentSetup(Protocol):
    def list_dataset_files(self) -> list[str]:
        pass

    def generate_ground_truth(self) -> dict[str, Any]:
        pass

    @property
    def output_directory(self) -> Path:
        return Path.cwd()


class MiniBuy:
    def __init__(self, data_dir: Path, prepare_matching: bool = True):
        self.input_directory = self.output_directory = data_dir
        self.__gen_input_file = self.input_directory / "Buy.csv"
        self.gold_standard = str(
            (self.output_directory / "Buy-ground-truth.json").absolute()
        )
        self.output_file = str((self.output_directory / "Buy-result.json").absolute())
        self.__prepare_matching = prepare_matching

    def generate_ground_truth(self) -> dict[str, Any]:
        if self.__prepare_matching:
            generate(
                str(self.__gen_input_file.absolute()),
                str(self.output_directory.absolute()),
                self.gold_standard,
                [
                    "name,manufacturer,price,id",
                    "description,name,id",
                ],
            )

        with open(self.gold_standard) as f:
            return orjson.loads(f.read())

    def list_dataset_files(self) -> list[str]:
        return [
            str((self.input_directory / f"{idx:05}-sub-Buy.csv").absolute())
            for idx in range(1, 3)
        ]

    def __str__(self):
        return "mini-buy"


class ExistingData:
    def __init__(
        self,
        data_dir: Path,
        ds1_name: str,
        ds2_name: str,
        perfect_mapping_name: str,
        ds1_pm_id_col: str,
        ds2_pm_id_col: str,
        prepare_matching: bool = False,
    ):
        self.__ds1_file = data_dir / ds1_name
        self.__ds2_file = data_dir / ds2_name
        self.__ideal_mapping_file = data_dir / perfect_mapping_name
        self.__gt_file = data_dir / "gt.json"
        self.output_directory = data_dir
        self.__prepare_matching = prepare_matching
        self._perf_mapping_id_col_1 = ds1_pm_id_col
        self._perf_mapping_id_col_2 = ds2_pm_id_col

    @staticmethod
    def __clean_input_data(value: Any) -> str:
        if isinstance(value, Number):
            if isnan(value):
                return ""
        if value is None:
            return ""
        return str(value)

    def generate_ground_truth(self) -> dict[str, Any]:
        if self.__prepare_matching:
            ds1 = pd.read_csv(
                self.__ds1_file, header=0, index_col="id", encoding_errors="ignore"
            ).applymap(self.__clean_input_data)
            ds2 = pd.read_csv(
                self.__ds2_file, header=0, index_col="id", encoding_errors="ignore"
            ).applymap(self.__clean_input_data)
            mapping = pd.read_csv(self.__ideal_mapping_file, header=0)
            pair_list = []
            input_set = {}
            for index, link in mapping.iterrows():
                ds1_id = link[self._perf_mapping_id_col_1]
                ds2_id = link[self._perf_mapping_id_col_2]
                ds1_ref = tuple(v for v in (*ds1.loc[ds1_id], str(ds1_id)))
                ds2_ref = tuple(v for v in (*ds2.loc[ds2_id], str(ds2_id)))
                pair = (ds1_ref, ds2_ref)
                pair_list.append(pair)
                input_set[ds1_ref] = None
                input_set[ds2_ref] = None

            gt = EntityResolutionResult()
            gt.fsm = pair_list
            gt.algebraic = compute_partition(list(input_set), pair_list)
            with open(self.__gt_file, "w") as f:
                f.write(orjson.dumps(gt).decode("utf-8"))

        with open(self.__gt_file, "r") as f:
            return orjson.loads(f.read())

    def list_dataset_files(self) -> list[str]:
        return [str(x.absolute()) for x in [self.__ds1_file, self.__ds2_file]]

    def __str__(self):
        return f"{self.__ds1_file.stem.lower()}-{self.__ds2_file.stem.lower()}"
