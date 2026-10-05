"""Une barre transitoire en terminal ; résumés seulement dans les logs."""
import sys
from tqdm.auto import tqdm


def message(text, verbose=True):
    if verbose:
        tqdm.write(text, file=sys.stderr)


def progress(items, description, verbose=True):
    return tqdm(items, desc=description, disable=not (verbose and sys.stderr.isatty()),
                leave=False, dynamic_ncols=True, mininterval=.3)
