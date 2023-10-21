import click

from ._entity_resolution import main as match_entities
from ._generate import main as generate
from ._compute_quality_metrics import main as compute_metrics

# from ._prepare_results import transform_results
import matchescu.cli._global_flags as flags


@click.group("matchescu")
@click.option("-v", "--verbose", type=click.BOOL, is_flag=True, default=False)
def main(verbose):
    flags.VERBOSE = verbose


main.add_command(generate, "generate")
main.add_command(match_entities, "entity-resolution")
main.add_command(compute_metrics, "compute-metrics")

main()
