import matchescu_cli._global_flags as flags


def _print(*args):
    if not flags.VERBOSE:
        return
    print(*args)


