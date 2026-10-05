# Ajouter et configurer un modèle

La boucle d'entraînement construit le modèle indiqué par `model`, puis lui
transmet des tenseurs `[B, C, size, size]`. Il doit renvoyer des **logits**
`[B, num_classes]`, sans softmax. `C=1` pour `raw`, `C=3` pour `pair` et
`multimodal` (radio, Chandra, masque).

## Utiliser ImageCNN

```json
{
  "name": "image_radio",
  "model": "image_cnn",
  "mode": "raw",
  "size": 128,
  "batch_size": 8,
  "model_params": {"hidden_dim": 64, "classifier_hidden_dim": 32}
}
```

Ce bloc se place dans `experiments` du JSON. Un exemple complet est fourni :

```sh
python -m clusterprep --list-models
python -m clusterprep clusterprep/examples/image_cnn.json
```

Les modèles intégrés sont `cnn`, `fusion`, `image_cnn`, `dual_encoder_cnn`,
`dual_ssn` et `dual_encoder_ssn`. `ImageCNN` accepte
les trois modes, avec concaténation des canaux en entrée. L'architecture de
ses couches est conservée. Son calcul initial des dimensions se fait en mode
évaluation pour ne pas modifier les statistiques BatchNorm.

`dual_encoder_cnn` accepte aussi les trois modes. Avec `"mode": "raw"`,
seul l'encodeur radio est construit et le classificateur reçoit son latent,
sans branche Chandra ni couverture. Avec `pair` ou `multimodal`, les deux
encodeurs sont utilisés. Les paramètres `use_mask` et `use_coverage`
concernent uniquement ces modes ; aucune option supplémentaire n'est requise
pour passer en `raw`.

## Utiliser DualSSN

Installer la dépendance optionnelle depuis la racine du projet :

```sh
pip install -e './clusterprep[scattering]'
```

`DualSSN` (`dual_ssn`) reprend les branches CNN image et scattering avec
attention SE du `DualScatterSqueezeNet` de `dcreclass`. Un seul encodeur
traite tous les canaux ensemble. `DualEncoderSSN` (`dual_encoder_ssn`)
utilise deux de ces encodeurs indépendants : radio et Chandra (+ masque
si `use_mask=true`). La fusion suit `DualEncoderCNN`, avec annulation du
latent Chandra lorsque la couverture est nulle et ajout optionnel de la
couverture au classificateur (`use_coverage=true`). En `raw`, seul
l'encodeur radio est construit. Les deux modèles acceptent les trois modes.

Le scattering Kymatio est calculé dans `forward`, canal par canal, sur les
images reçues après augmentation/MixUp. Aucun tenseur scattering séparé ni
cache n'est nécessaire. Les filtres d'ondelettes sont fixes ; les branches
CNN, l'attention et le classificateur sont entraînables. Les checkpoints
incluent les filtres. Il n'y a pas de normalisation globale supplémentaire
des coefficients, contrairement aux configurations de `dcreclass` qui en
appliquent une ; ses checkpoints ne sont pas directement interchangeables.

Exemple à placer dans `experiments` :

```json
{
  "name": "dual_ssn_multicanal",
  "model": "dual_ssn",
  "mode": "multimodal",
  "device": "cpu",
  "size": 128,
  "batch_size": 8,
  "model_params": {
    "J": 2, "L": 8, "max_order": 2,
    "hidden_dim1": 32, "hidden_dim2": 16,
    "classifier_hidden_dim": 32, "dropout_rate": 0.5
  }
}
```

Pour deux encodeurs, utiliser `"model": "dual_encoder_ssn"` et par exemple :

```json
"model_params": {
  "J": 2, "L": 8, "max_order": 2, "hidden_dim2": 16,
  "fusion_hidden_dim": 64, "dropout_rate": 0.5,
  "use_mask": true, "use_coverage": true
}
```

`J` va de 1 à 4 et impose `size >= 2**J`, `L` est le nombre d'orientations,
et `max_order` vaut 1 ou 2. `hidden_dim2` règle la largeur de la branche
scattering. Le calcul du scattering à chaque passage augmente le coût
par rapport aux CNN seuls, particulièrement avec deux encodeurs.

## Ajouter une architecture dans models.py

Définir une classe PyTorch et lui donner un nom via le décorateur :

```python
@register_model("my_cnn")
class MyCNN(nn.Module):
    def __init__(self, in_channels, num_classes, features=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(features, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
```

La configuration devient `"model": "my_cnn"` et
`"model_params": {"features": 64}`. Il n'y a aucune branche à ajouter dans
`build_model` ou dans le moteur d'entraînement. Pour restreindre les entrées,
utiliser `@register_model("my_fusion", modes=("pair", "multimodal"))`.
Un nom déjà utilisé est refusé plutôt qu'écrasé.

## Garder les modèles dans un module externe

Une classe peut aussi rester dans votre propre package importable :

```json
{
  "name": "custom",
  "model": "my_project.models:MyCNN",
  "model_params": {"features": 64}
}
```

Le chemin `module:Constructeur` importe directement la classe ou une fonction
de construction ; aucun décorateur ni modification de clusterprep n'est
nécessaire. Le module doit être installé ou accessible dans `PYTHONPATH`, y
compris au rechargement du checkpoint. Si vous utilisez à la place un nom
enregistré par un décorateur dans un module externe, importez ce module avant
d'appeler `build_model` ; le chemin explicite est plus simple pour la CLI.

## Arguments injectés et validation

Le constructeur reçoit automatiquement les arguments suivants **s'il les
déclare explicitement dans sa signature** :

| Argument | Origine |
| --- | --- |
| `in_channels` | 1 ou 3 selon le mode |
| `input_shape` | `(in_channels, size, size)` |
| `num_classes` | Nombre de classes configurées |
| `width` | `config.width`, sauf surcharge dans `model_params` |

Les autres arguments proviennent de `model_params` ou des valeurs par défaut
du constructeur. Les dimensions et le nombre de classes ne peuvent pas être
remplacés dans `model_params`. Une signature `**kwargs` seule ne reçoit pas
les arguments automatiques : les déclarer explicitement si nécessaires.
Les paramètres manquants/inconnus et les modes incompatibles sont signalés
avant le prétraitement FITS. Les paramètres doivent être sérialisables en JSON.

`model_params` est enregistré dans les configurations et les checkpoints :

```python
checkpoint = torch.load("best.pt", map_location="cpu", weights_only=True)
config = ExperimentConfig(**checkpoint["config"])
model = build_model(config, len(checkpoint["class_names"]))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

L'API existante `model_factory(config, num_classes)` reste disponible pour les
cas plus spécifiques. Elle remplace la construction par registre pour cet appel.

## BatchNorm et lots incomplets

Pour les modèles contenant BatchNorm, `batch_size >= 2` est requis. Si le
dernier lot train n'a qu'un exemple, il est fusionné avec le précédent : le
dernier lot peut donc contenir `batch_size + 1` exemples, sans perte de données.
La validation et le test utilisent `eval()` et conservent leurs lots habituels.
