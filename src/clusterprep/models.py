"""Petits modèles de référence ; les expériences acceptent aussi une factory."""
import importlib
import inspect
import torch
from torch import nn

_MODELS = {}


def register_model(name, *, modes=("raw", "pair", "multimodal")):
    """Décorateur pour un constructeur nn.Module ; refuse les noms déjà utilisés."""
    if not isinstance(name, str) or not name or ":" in name:
        raise ValueError("Nom de modèle non vide et sans ':' requis")
    modes = frozenset(modes)
    if not modes or not modes <= {"raw", "pair", "multimodal"}:
        raise ValueError("Modes de données invalides")

    def decorate(constructor):
        if name in _MODELS:
            raise ValueError(f"Modèle déjà enregistré : {name}")
        if not callable(constructor):
            raise TypeError("Le constructeur doit être appelable")
        _MODELS[name] = (constructor, modes)
        return constructor
    return decorate


def list_models():
    """Noms enregistrés et modes supportés."""
    return {name: sorted(modes) for name, (_, modes) in sorted(_MODELS.items())}


def _encoder(channels, width):
    return nn.Sequential(
        nn.Conv2d(channels, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())

class ImageEncoder(nn.Module):
    """
    Encode une image [B, C, H, W] vers un vecteur latent [B, latent_dim].
    """

    def __init__(self, input_shape):
        super().__init__()

        in_channels, H, W = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),
        )

        self.to_latent = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
        )

        # Détermination automatique de la taille du latent
        self.encoder.eval()
        self.to_latent.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)
            out = self.to_latent(self.encoder(dummy))
            self.output_dim = out.flatten(1).shape[1]

        self.encoder.train()
        self.to_latent.train()

    def forward(self, x):
        x = self.encoder(x)
        x = self.to_latent(x)
        return x.flatten(1)


@register_model("cnn")
class SimpleCNN(nn.Module):
    """Fusion précoce : un canal radio ou radio/Chandra/masque concaténés."""
    def __init__(self, in_channels=1, num_classes=2, width=16):
        super().__init__()
        self.encoder = _encoder(in_channels, width)
        self.classifier = nn.Linear(width * 2, num_classes)

    def forward(self, x):
        return self.classifier(self.encoder(x))


@register_model("fusion", modes=("pair", "multimodal"))
class FusionCNN(nn.Module):
    """Deux branches ; la branche X est annulée si sa couverture est nulle."""
    def __init__(self, num_classes=2, width=16):
        super().__init__()
        self.radio = _encoder(1, width)
        self.chandra = _encoder(2, width)
        self.classifier = nn.Linear(width * 4 + 1, num_classes)

    def forward(self, x):
        raw_features = self.radio(x[:, :1])
        coverage = x[:, 2:3].mean(dim=(2, 3))
        x_features = self.chandra(x[:, 1:3]) * (coverage > 0).to(x.dtype)
        return self.classifier(torch.cat([raw_features, x_features, coverage], dim=1))

@register_model("image_cnn")
class ImageCNN(nn.Module):

    def __init__(
        self,
        input_shape,
        num_classes=2,
        hidden_dim=32,
        classifier_hidden_dim=32,
    ):
        super().__init__()

        self.encoder = ImageEncoder(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(hidden_dim, classifier_hidden_dim),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

@register_model("dual_encoder_cnn")
class DualEncoderCNN(nn.Module):
    """Encodeur radio seul en raw, fusion radio/Chandra dans les autres modes."""

    def __init__(
        self,
        input_shape,
        num_classes=2,
        fusion_hidden_dim=64,
        use_mask=True,
        use_coverage=True,
    ):
        super().__init__()

        in_channels, H, W = input_shape
        if in_channels not in (1, 3):
            raise ValueError("DualEncoderCNN attend 1 canal (raw) ou 3 canaux (radio/Chandra/masque)")

        self.use_chandra = in_channels == 3
        self.use_mask = use_mask
        self.use_coverage = use_coverage and self.use_chandra

        # RAW uniquement
        self.radio_encoder = ImageEncoder(
            input_shape=(1, H, W)
        )

        # CHANDRA + mask
        chandra_channels = 2 if use_mask else 1

        self.chandra_encoder = ImageEncoder(
            input_shape=(chandra_channels, H, W)
        ) if self.use_chandra else None

        fusion_dim = self.radio_encoder.output_dim
        if self.use_chandra:
            fusion_dim += self.chandra_encoder.output_dim

        if self.use_coverage:
            fusion_dim += 1

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, x):

        # x : [B, 1, H, W] en raw, sinon [B, 3, H, W]
        radio = x[:, 0:1]
        chandra = x[:, 1:2]
        mask = x[:, 2:3]

        # -------------------------
        # Radio encoder
        # -------------------------
        radio_features = self.radio_encoder(radio)
        if not self.use_chandra:
            return self.classifier(radio_features)

        # -------------------------
        # Chandra encoder
        # -------------------------
        if self.use_mask:
            chandra_input = torch.cat(
                [chandra, mask],
                dim=1,
            )
        else:
            chandra_input = chandra

        chandra_features = self.chandra_encoder(
            chandra_input
        )

        # Fraction de l'image couverte par Chandra
        coverage = mask.mean(dim=(2, 3))

        # Si aucune donnée Chandra :
        # pas de contribution du X-ray
        has_chandra = (coverage > 0).to(chandra_features.dtype)

        chandra_features = (
            chandra_features * has_chandra
        )

        # -------------------------
        # Fusion
        # -------------------------
        features = [
            radio_features,
            chandra_features,
        ]

        if self.use_coverage:
            features.append(coverage)

        features = torch.cat(features, dim=1)

        return self.classifier(features)


class SEBlock(nn.Module):
    """Attention par canal (squeeze-and-excitation), comme dans dcreclass."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DualSSNEncoder(nn.Module):
    """Latent image + scattering de DualSSN, calculé depuis [B, C, H, W].

    Le scattering fixe est appliqué séparément à chaque canal, puis les axes
    canal/coefficient sont fusionnés avant les convolutions apprenables.
    """

    def __init__(self, input_shape, hidden_dim2=16, J=2, L=8, max_order=2):
        super().__init__()
        if isinstance(J, bool) or not isinstance(J, int) or J not in (1, 2, 3, 4):
            raise ValueError("J doit être 1, 2, 3 ou 4")
        if isinstance(L, bool) or not isinstance(L, int) or L < 1:
            raise ValueError("L doit être un entier positif")
        if isinstance(max_order, bool) or max_order not in (1, 2):
            raise ValueError("max_order doit être 1 ou 2")
        if not isinstance(hidden_dim2, int) or hidden_dim2 < 1:
            raise ValueError("hidden_dim2 doit être un entier positif")
        channels, height, width = input_shape
        if channels < 1 or min(height, width) < 2 ** J:
            raise ValueError("input_shape requiert C >= 1 et H, W >= 2**J")
        try:
            from kymatio.torch import Scattering2D
        except ImportError as error:
            raise ImportError(
                "DualSSN nécessite kymatio : pip install 'clusterprep[scattering]'"
            ) from error
        self.input_shape = tuple(input_shape)
        self.scattering = Scattering2D(J=J, shape=(height, width), L=L, max_order=max_order)

        def block(in_ch, out_ch, kernel=3, stride=1, dropout=0.2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2), nn.Dropout2d(dropout),
            )

        self.cnn_encoder = nn.Sequential(
            block(channels, 8, kernel=5), block(8, 16), nn.MaxPool2d(2),
            block(16, 32, kernel=5, dropout=0.3),
        )
        self.conv_to_latent_img = nn.Sequential(*[
            block(32, 32, stride=2, dropout=0.4 if i == 3 else 0.3)
            for i in range(4)
        ])
        from kymatio.torch import Scattering2D
        temporary_scattering = Scattering2D(
            J=J,
            shape=(height, width),
            L=L,
            max_order=max_order,
        )
        with torch.no_grad():
            dummy = torch.zeros(
                1,
                channels,
                height,
                width,
            )
            scat = temporary_scattering(dummy).flatten(1, 2)
        scat_channels = scat.shape[1]
        del temporary_scattering
        scat_blocks = [
            block(
                scat_channels,
                hidden_dim2,
            ),
            SEBlock(hidden_dim2),
        ]
        for i in range(5 - J):
            scat_blocks.extend([
                block(hidden_dim2, hidden_dim2, dropout=0.2 + i * 0.1),
                SEBlock(hidden_dim2),
                block(hidden_dim2, hidden_dim2, stride=2, dropout=0.2 + i * 0.1),
                SEBlock(hidden_dim2),
            ])
        self.conv_to_latent_scat = nn.Sequential(*scat_blocks)
        # Aucun compteur ni statistique BatchNorm modifié à la construction.
        self.eval()
        with torch.no_grad():
            img_features = self.conv_to_latent_img(self.cnn_encoder(dummy))
            scat_features = self.conv_to_latent_scat(scat)
            self.output_dim = img_features.numel() + scat_features.numel()
        self.train()

    def forward(self, x, scattering):
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"DualSSN attend [B, {self.input_shape}]"
            )
        # branche CNN classique
        img = self.conv_to_latent_img(
            self.cnn_encoder(x)
        ).flatten(1)
        # scattering déjà calculé :
        #
        # [B,C,K,Hs,Ws]
        #       ↓
        # [B,C*K,Hs,Ws]
        if scattering.ndim != 5:
            raise ValueError(
                "Scattering attendu sous forme [B,C,K,Hs,Ws]"
            )
        scat = scattering.flatten(1, 2)
        scat = self.conv_to_latent_scat(
            scat
        ).flatten(1)
        return torch.cat(
            [img, scat],
            dim=1,
        )


@register_model("dual_ssn")
class DualSSN(nn.Module):
    """Un encodeur DualSSN multicanal : fusion précoce radio/Chandra/masque."""

    uses_precomputed_scattering = True

    def __init__(
        self,
        input_shape,
        num_classes=2,
        hidden_dim1=32,
        hidden_dim2=16,
        classifier_hidden_dim=32,
        dropout_rate=0.5,
        J=2,
        L=8,
        max_order=2,
    ):
        super().__init__()

        self.scattering_config = {
            "J": J,
            "L": L,
            "max_order": max_order,
        }

        self.encoder = DualSSNEncoder(
            input_shape,
            hidden_dim2,
            J,
            L,
            max_order,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim1), nn.BatchNorm1d(hidden_dim1),
            nn.LeakyReLU(0.2), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, classifier_hidden_dim), nn.BatchNorm1d(classifier_hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x, scattering):
        return self.classifier(
            self.encoder(x, scattering)
        )


@register_model("dual_encoder_ssn")
class DualEncoderSSN(nn.Module):
    """Deux encodeurs DualSSN indépendants, fusionnés comme DualEncoderCNN.

    En raw, seul l'encodeur radio est construit. Sinon, le latent Chandra
    (image et scattering) est annulé lorsque la couverture est nulle.
    """

    uses_precomputed_scattering = True

    def __init__(self, input_shape, num_classes=2, fusion_hidden_dim=64,
                 hidden_dim2=16, use_mask=True, use_coverage=True,
                 dropout_rate=0.5, J=2, L=8, max_order=2):
        super().__init__()
        self.scattering_config = {
            "J": J,
            "L": L,
            "max_order": max_order,
        }
        channels, height, width = input_shape
        if channels not in (1, 3):
            raise ValueError("DualEncoderSSN attend 1 canal (raw) ou 3 canaux (radio/Chandra/masque)")
        self.input_shape = tuple(input_shape)
        self.use_chandra = channels == 3
        self.use_mask = use_mask
        self.use_coverage = use_coverage and self.use_chandra
        self.radio_encoder = DualSSNEncoder((1, height, width), hidden_dim2, J, L, max_order)
        self.chandra_encoder = (
            DualSSNEncoder((2 if use_mask else 1, height, width), hidden_dim2, J, L, max_order)
            if self.use_chandra else None
        )
        fusion_dim = self.radio_encoder.output_dim
        if self.use_chandra:
            fusion_dim += self.chandra_encoder.output_dim
        fusion_dim += int(self.use_coverage)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim), nn.BatchNorm1d(fusion_hidden_dim),
            nn.LeakyReLU(0.2), nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, x, scattering):

        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"DualEncoderSSN attend [B, {self.input_shape}]"
            )

        # scattering :
        # [B,C,K,Hs,Ws]

        radio_scattering = scattering[:, 0:1]

        features = [
            self.radio_encoder(
                x[:, :1],
                radio_scattering,
            )
        ]

        if self.use_chandra:

            coverage = x[:, 2:3].mean(
                dim=(2, 3)
            )

            if self.use_mask:
                chandra = x[:, 1:3]
                chandra_scattering = scattering[:, 1:3]

            else:
                chandra = x[:, 1:2]
                chandra_scattering = scattering[:, 1:2]

            chandra_features = self.chandra_encoder(
                chandra,
                chandra_scattering,
            )

            chandra_features = (
                chandra_features
                * (coverage > 0).to(
                    chandra_features.dtype
                )
            )

            features.append(chandra_features)

            if self.use_coverage:
                features.append(coverage)

        return self.classifier(
            torch.cat(features, dim=1)
        )

def _list_models_with_scattering():
    """Noms de modèles utilisant le scattering, pour avertir sur mixup."""
    return [name for name, (constructor, _) in _MODELS.items()
            if inspect.isclass(constructor) and issubclass(constructor, DualSSN)]

def build_model(config, num_classes, input_shape=None):
    """Construit un modèle enregistré ou importable via 'module:Constructeur'.

    Les arguments explicitement déclarés input_shape, in_channels, num_classes
    et width sont injectés. Les autres proviennent de config.model_params.
    """
    name = config.model
    if name in _MODELS:
        constructor, modes = _MODELS[name]
        if config.mode not in modes:
            raise ValueError(f"{name} ne supporte pas le mode {config.mode} ; modes : {sorted(modes)}")
    elif ":" in name:
        module, attribute = name.split(":", 1)
        try:
            constructor = getattr(importlib.import_module(module), attribute)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Impossible d'importer le modèle {name}") from error
    else:
        raise ValueError(f"Modèle inconnu : {name}. Disponibles : {', '.join(list_models())}. "
                         "Ou utiliser module:Constructeur.")
    expected_shape = (1 if config.mode == "raw" else 3, config.size, config.size)
    if input_shape is not None and tuple(input_shape) != expected_shape:
        raise ValueError(f"input_shape doit correspondre aux données : {expected_shape}")
    parameters = dict(getattr(config, "model_params", {}))
    if {"input_shape", "in_channels", "num_classes"} & parameters.keys():
        raise ValueError("input_shape, in_channels et num_classes sont déduits des données")
    signature = inspect.signature(constructor)
    automatic = {"input_shape": expected_shape, "in_channels": expected_shape[0],
                 "num_classes": num_classes, "width": config.width}
    for key, value in automatic.items():
        if key in signature.parameters:
            parameters.setdefault(key, value)
    try:
        signature.bind(**parameters)
    except TypeError as error:
        raise ValueError(f"Paramètres invalides pour {name} : {error}") from error
    model = constructor(**parameters)
    if not isinstance(model, nn.Module):
        raise TypeError(f"{name} doit construire un torch.nn.Module")
    return model
