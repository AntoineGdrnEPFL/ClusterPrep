"""Optional notebook controls, with configuration-cell fallback."""
from .registry import registry_table


def filter_controls(runs, defaults=None):
    """Return (widget, read_filters); changing controls never silently reruns test analysis."""
    import ipywidgets as widgets
    defaults = defaults or {}
    table = registry_table(runs)
    controls = {}
    for key in ('task_type', 'task', 'model', 'noise_method', 'modalities', 'experiment_id', 'fold', 'transfer_strategy', 'configuration_id'):
        choices = sorted(table[key].dropna().unique(), key=str) if key in table else []
        default = defaults.get(key)
        values = list(default) if isinstance(default, (list, tuple, set)) else ([default] if default is not None else [])
        controls[key] = widgets.SelectMultiple(options=[(str(table.loc[table.configuration_id == v, 'configuration_name'].iloc[0]) if key == 'configuration_id' else str(v), v) for v in choices],
                                               value=tuple(v for v in values if v in choices),
                                               description=key, rows=min(5, max(1, len(choices))),
                                               style={'description_width':'130px'},
                                               layout=widgets.Layout(width='600px'))
    box = widgets.VBox([widgets.HTML('Sélection vide = tous. Après modification, réexécuter la cellule « Appliquer » et les suivantes.'), *controls.values()])
    return box, lambda: {key: list(control.value) or None for key, control in controls.items()}


def display_table(table):
    """Human-readable notebook view; exports retain the unique join keys."""
    aliases = {'run_id': 'run_name', 'run_ids': 'run_names',
               'configuration_id': 'configuration_name',
               'left': 'left_name', 'right': 'right_name'}
    if any(key in table.index.names for key in aliases):
        table = table.reset_index()
    return table.drop(columns=[key for key, name in aliases.items()
                               if key in table and name in table])
