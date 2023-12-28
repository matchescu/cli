import json
from dataclasses import asdict
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
            return json.load(f)

    def list_dataset_files(self) -> list[str]:
        return [
            str((self.input_directory / f"{idx:05}-sub-Buy.csv").absolute())
            for idx in range(1, 3)
        ]


class AbtBuy:
    def __init__(self, data_dir: Path, prepare_matching: bool = False):
        self.__abt_file = data_dir / "Abt.csv"
        self.__buy_file = data_dir / "Buy.csv"
        self.__ideal_mapping_file = data_dir / "abt_buy_perfectMapping.csv"
        self.__gt_file = data_dir / "gt.json"
        self.output_directory = data_dir
        self.__prepare_matching = prepare_matching

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
            abt = pd.read_csv(
                self.__abt_file, header=0, index_col="id", encoding_errors="ignore"
            ).applymap(self.__clean_input_data)
            buy = pd.read_csv(
                self.__buy_file, header=0, index_col="id", encoding_errors="ignore"
            ).applymap(self.__clean_input_data)
            mapping = pd.read_csv(self.__ideal_mapping_file, header=0)
            pair_list = []
            input_set = {}
            for index, link in mapping.iterrows():
                id_abt = link["idAbt"]
                id_buy = link["idBuy"]
                abt_ref = tuple(v for v in (*abt.loc[id_abt], str(id_abt)))
                buy_ref = tuple(v for v in (*buy.loc[id_buy], str(id_buy)))
                pair = (abt_ref, buy_ref)
                pair_list.append(pair)
                input_set[abt_ref] = None
                input_set[buy_ref] = None

            gt = EntityResolutionResult()
            gt.fsm = pair_list
            gt.algebraic = compute_partition(list(input_set), pair_list)
            obj = asdict(gt)
            with open(self.__gt_file, "w") as f:
                json.dump(obj, f, indent=4)

        with open(self.__gt_file, "r") as f:
            return json.load(f)

    def list_dataset_files(self) -> list[str]:
        return [str(x.absolute()) for x in [self.__abt_file, self.__buy_file]]
