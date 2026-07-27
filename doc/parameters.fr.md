# Référence des paramètres QuEst

Cette page est générée à partir de `parameter_catalog.json`, `quest/settings/locales/fr.json`. Modifiez-les d'abord pour changer le texte d'aide d'un paramètre — l'interface web, cette page et le formulaire de bureau lisent les mêmes fichiers.

## Résumé du catalogue

| Clé | Catégorie | Type | Unité | Description |
| --- | --- | --- | --- | --- |
| `amino_acid_interactions` | Extinction | table | - | Table d'interaction par type de résidu. Chaque entrée contient slow_factor (0–1, mesure du ralentissement du fluorophore par le résidu), kQ (taux d'extinction en 1/ns appliqué au contact), quench_radius (rayon de contact en ångströms depuis le centre du fluorophore ; null hérite de critical_distance) et quench_atoms (noms des atomes dont le barycentre définit le centre d'extinction). Par défaut, seuls les résidus à chaîne latérale rédox-active éteignent la fluorescence : TRP, TYR, MET, HIS, CYS et PRO. |
| `amino_acid_interactions.kQ` | Extinction | number | 1/ns | Taux auquel ce type de résidu éteint le fluorophore tant que les deux sont en contact. Les contributions des résidus dont les sphères de contact se recouvrent s'additionnent. |
| `amino_acid_interactions.quench_atoms` | Extinction | list | - | Noms des atomes dont le barycentre définit le centre d'extinction de ce résidu. Les valeurs par défaut suivent le groupement rédox-actif qui assure le transfert d'électron photo-induit : le cycle indole pour TRP, le cycle phénol pour TYR, le cycle imidazole pour HIS, SD pour MET et SG pour CYS — et non CB. |
| `amino_acid_interactions.quench_radius` | Extinction | number | Å | Distance entre le centre du fluorophore et le centre d'extinction de ce résidu en deçà de laquelle l'extinction se produit. Laisser vide pour hériter du critical_distance du projet. |
| `amino_acid_interactions.slow_factor` | Extinction | number | - | Facteur compris entre 0 et 1 qui met à l'échelle le coefficient de diffusion du fluorophore au voisinage de ce type de résidu, modélisant une adhérence non spécifique. Les résidus qui se recouvrent multiplient leurs facteurs. |
| `attachment.atom` | Site de marquage | text | - | Nom de l'atome (par ex. CB, CA) servant de point d'attache au fluorophore. |
| `attachment.chain` | Site de marquage | text | - | Identifiant de chaîne du résidu sur lequel le fluorophore est attaché. |
| `attachment.residue` | Site de marquage | integer | - | Numéro de séquence du résidu sur lequel le fluorophore est attaché. |
| `av_parameter.linker_length` | Volume accessible | number | Å | Distance maximale entre le point d'attache et le centre du fluorophore. |
| `av_parameter.linker_width` | Volume accessible | number | Å | Largeur du bras espaceur qui retient le fluorophore. |
| `av_parameter.radius1` | Volume accessible | number | Å | Rayon d'exclusion représentant le fluorophore comme une sphère (approximation à une sphère). |
| `coarse_grained` | Simulation | boolean | - | Réduit la structure au squelette et aux CB avant la simulation, chaque CB étant déplacé au centre de masse de sa chaîne latérale. Les atomes de chaîne latérale sont alors absents : le centre d'extinction de chaque résidu se rabat sur ce pseudo-atome CB, et le volume accessible augmente puisque les chaînes latérales ne l'obstruent plus. C'est un choix de modélisation, non une optimisation : le coût de QuEst tient à la grille du volume accessible, pas au nombre d'atomes. |
| `critical_distance` | Extinction | number | Å | Rayon de contact employé par les types de résidus qui ne définissent pas leur propre quench_radius. Il se mesure du centre du fluorophore au centre d'extinction du résidu et doit donc dépasser le rayon du fluorophore augmenté du contact de van der Waals (typiquement 7 à 9 Å pour un fluorophore de 3,5 Å). |
| `dg` | Volume accessible | number | Å | Résolution de la grille utilisée pour calculer le volume accessible (AV) et l'espace de diffusion. |
| `fret.R0_matrix` | Accepteur FRET | number | Å | Rayon de Förster R0 du couple donneur–accepteur, en ångströms comme toute autre longueur du projet. Il est stocké dans la case donneur→accepteur de fret.R0_matrix ; le taux de transfert varie en (R0/r)^6, si bien qu'une saisie en nanomètres fait disparaître le FRET. Les couples xanthènes usuels se situent entre 45 et 65 Å. |
| `fret.acceptor_dynamics` | Accepteur FRET | choice | - | Traitement de l'accepteur : « trajectory » simule sa marche brownienne avec son propre coefficient de diffusion et sa propre adhérence, et utilise la distance donneur-accepteur instantanée ; « averaged » moyenne sur son volume accessible statique (limite de l'accepteur rapide, valable seulement s'il parcourt ce volume rapidement devant la durée de vie du donneur). Les deux diffèrent parce qu'un nuage est pondéré uniformément alors qu'une trajectoire l'est par les endroits où le fluorophore séjourne. |
| `fret.dyes.0.D` | Simulation | number | Å²/ns | Coefficient de diffusion du donneur en solution, hors interaction avec la surface de la protéine. |
| `fret.dyes.1.D` | Accepteur FRET | number | Å²/ns | Coefficient de diffusion de l'accepteur FRET hors interaction avec la surface de la protéine. |
| `fret.dyes.1.amino_acid_interactions` | Accepteur FRET | table | - | Table d'interaction par type de résidu pour l'accepteur, de même forme que l'amino_acid_interactions du donneur. Seul slow_factor est utilisé, car QuEst rapporte le déclin du donneur : l'extinction PET propre à l'accepteur modifierait sa brillance, pas celle du donneur. Par défaut, la table du donneur. |
| `fret.dyes.1.attachment.atom` | Accepteur FRET | text | - | Nom de l'atome sur lequel l'accepteur FRET est attaché. |
| `fret.dyes.1.attachment.chain` | Accepteur FRET | text | - | Identifiant de chaîne du résidu sur lequel l'accepteur FRET est attaché. |
| `fret.dyes.1.attachment.residue` | Accepteur FRET | integer | - | Numéro de séquence du résidu sur lequel l'accepteur FRET est attaché. |
| `fret.dyes.1.av_parameter.linker_length` | Accepteur FRET | number | Å | Distance maximale entre le point d'attache et le centre de l'accepteur FRET. |
| `fret.dyes.1.av_parameter.linker_width` | Accepteur FRET | number | Å | Largeur du bras espaceur qui retient l'accepteur FRET. |
| `fret.dyes.1.av_parameter.radius1` | Accepteur FRET | number | Å | Rayon d'exclusion représentant l'accepteur FRET comme une sphère. |
| `fret.dyes.1.slow_radius` | Accepteur FRET | number | Å | Rayon autour de chaque centre d'acide aminé à l'intérieur duquel ce résidu ralentit l'accepteur. Distinct de celui du donneur : une cyanine et une rhodamine n'adhèrent pas de la même façon à une surface protéique. Par défaut, la valeur du donneur. |
| `fret.enabled` | Accepteur FRET | boolean | - | Indique si un accepteur FRET est simulé afin de calculer l'efficacité de FRET et les courbes de déclin. |
| `n_bins` | Simulation | integer | - | Nombre de canaux temporels de l'histogramme de déclin produit. |
| `n_photons` | Simulation | integer | - | Nombre de photons d'excitation simulés pour reconstruire l'histogramme de déclin de fluorescence. |
| `output_file` | Avancé | text | - | Préfixe de nom employé pour enregistrer les résultats, les trajectoires et les déclins. |
| `parallel_trajectories` | Avancé | integer | - | Nombre de trajectoires browniennes indépendantes calculées simultanément et réunies avant l'échantillonnage des photons. −1 utilise tous les cœurs du processeur, dans la limite de 8. |
| `pdb` | Structure | file | - | Chemin du fichier de structure (PDB/CIF), ou identifiant RCSB PDB à quatre caractères, utilisé pour le calcul du volume accessible. |
| `random_seed` | Avancé | integer | - | Graine de la simulation Monte-Carlo. Lorsqu'elle est fixée, la trajectoire de diffusion et le tirage des photons sont tous deux initialisés : deux exécutions du même projet reproduisent exactement le même déclin et le même rendement quantique. Sans graine, chaque exécution est indépendante et le tirage des photons emploie un noyau multithread plus rapide. |
| `save_avs` | Avancé | boolean | - | Indique si les grilles de volume accessible calculées sont enregistrées au format XYZ. |
| `skip_frame` | Avancé | integer | - | Facteur de sous-échantillonnage lors de l'écriture des images de trajectoire dans les fichiers de coordonnées. |
| `slow_radius` | Avancé | number | Å | Rayon autour de chaque centre d'acide aminé de la protéine entière à l'intérieur duquel ce type de résidu applique son slow_factor à la grille des coefficients de diffusion. |
| `t_max` | Simulation | number | ns | Durée totale de la trajectoire de dynamique brownienne simulée. |
| `t_step` | Simulation | number | ns | Pas de temps du calcul de la trajectoire de dynamique brownienne. |
| `tau0` | Simulation | number | ns | Durée de vie de fluorescence du donneur en l'absence d'extinction. |

## Structure

### `pdb`

**Libellé:** Fichier de structure  
**Type:** `file`

Chemin du fichier de structure (PDB/CIF), ou identifiant RCSB PDB à quatre caractères, utilisé pour le calcul du volume accessible.

## Site de marquage

### `attachment.atom`

**Libellé:** Atome d'attache  
**Type:** `text`

Nom de l'atome (par ex. CB, CA) servant de point d'attache au fluorophore.

### `attachment.chain`

**Libellé:** Chaîne d'attache  
**Type:** `text`

Identifiant de chaîne du résidu sur lequel le fluorophore est attaché.

### `attachment.residue`

**Libellé:** Résidu d'attache  
**Type:** `integer`

Numéro de séquence du résidu sur lequel le fluorophore est attaché.

## Volume accessible

### `av_parameter.linker_length`

**Libellé:** Longueur du bras espaceur  
**Type:** `number`
**Unité:** `Å`

Distance maximale entre le point d'attache et le centre du fluorophore.

### `av_parameter.linker_width`

**Libellé:** Largeur du bras espaceur  
**Type:** `number`
**Unité:** `Å`

Largeur du bras espaceur qui retient le fluorophore.

### `av_parameter.radius1`

**Libellé:** Rayon du fluorophore  
**Type:** `number`
**Unité:** `Å`

Rayon d'exclusion représentant le fluorophore comme une sphère (approximation à une sphère).

### `dg`

**Libellé:** Résolution de la grille  
**Type:** `number`
**Unité:** `Å`

Résolution de la grille utilisée pour calculer le volume accessible (AV) et l'espace de diffusion.

## Simulation

### `coarse_grained`

**Libellé:** Structure gros grains  
**Type:** `boolean`

Réduit la structure au squelette et aux CB avant la simulation, chaque CB étant déplacé au centre de masse de sa chaîne latérale. Les atomes de chaîne latérale sont alors absents : le centre d'extinction de chaque résidu se rabat sur ce pseudo-atome CB, et le volume accessible augmente puisque les chaînes latérales ne l'obstruent plus. C'est un choix de modélisation, non une optimisation : le coût de QuEst tient à la grille du volume accessible, pas au nombre d'atomes.

### `fret.dyes.0.D`

**Libellé:** Coefficient de diffusion du donneur  
**Type:** `number`
**Unité:** `Å²/ns`

Coefficient de diffusion du donneur en solution, hors interaction avec la surface de la protéine.

### `n_bins`

**Libellé:** Canaux du déclin  
**Type:** `integer`

Nombre de canaux temporels de l'histogramme de déclin produit.

### `n_photons`

**Libellé:** Photons simulés  
**Type:** `integer`

Nombre de photons d'excitation simulés pour reconstruire l'histogramme de déclin de fluorescence.

### `t_max`

**Libellé:** Durée de simulation  
**Type:** `number`
**Unité:** `ns`

Durée totale de la trajectoire de dynamique brownienne simulée.

### `t_step`

**Libellé:** Pas de temps  
**Type:** `number`
**Unité:** `ns`

Pas de temps du calcul de la trajectoire de dynamique brownienne.

### `tau0`

**Libellé:** Durée de vie non éteinte  
**Type:** `number`
**Unité:** `ns`

Durée de vie de fluorescence du donneur en l'absence d'extinction.

## Avancé

### `output_file`

**Libellé:** Préfixe des fichiers de sortie  
**Type:** `text`

Préfixe de nom employé pour enregistrer les résultats, les trajectoires et les déclins.

### `parallel_trajectories`

**Libellé:** Trajectoires parallèles  
**Type:** `integer`

Nombre de trajectoires browniennes indépendantes calculées simultanément et réunies avant l'échantillonnage des photons. −1 utilise tous les cœurs du processeur, dans la limite de 8.

### `random_seed`

**Libellé:** Graine aléatoire  
**Type:** `integer`

Graine de la simulation Monte-Carlo. Lorsqu'elle est fixée, la trajectoire de diffusion et le tirage des photons sont tous deux initialisés : deux exécutions du même projet reproduisent exactement le même déclin et le même rendement quantique. Sans graine, chaque exécution est indépendante et le tirage des photons emploie un noyau multithread plus rapide.

### `save_avs`

**Libellé:** Enregistrer les fichiers AV  
**Type:** `boolean`

Indique si les grilles de volume accessible calculées sont enregistrées au format XYZ.

### `skip_frame`

**Libellé:** Images de trajectoire ignorées  
**Type:** `integer`

Facteur de sous-échantillonnage lors de l'écriture des images de trajectoire dans les fichiers de coordonnées.

### `slow_radius`

**Libellé:** Rayon de ralentissement  
**Type:** `number`
**Unité:** `Å`

Rayon autour de chaque centre d'acide aminé de la protéine entière à l'intérieur duquel ce type de résidu applique son slow_factor à la grille des coefficients de diffusion.

## Accepteur FRET

### `fret.R0_matrix`

**Libellé:** Rayon de Förster  
**Type:** `number`
**Unité:** `Å`

Rayon de Förster R0 du couple donneur–accepteur, en ångströms comme toute autre longueur du projet. Il est stocké dans la case donneur→accepteur de fret.R0_matrix ; le taux de transfert varie en (R0/r)^6, si bien qu'une saisie en nanomètres fait disparaître le FRET. Les couples xanthènes usuels se situent entre 45 et 65 Å.

### `fret.acceptor_dynamics`

**Libellé:** Dynamique de l'accepteur  
**Type:** `choice`

Traitement de l'accepteur : « trajectory » simule sa marche brownienne avec son propre coefficient de diffusion et sa propre adhérence, et utilise la distance donneur-accepteur instantanée ; « averaged » moyenne sur son volume accessible statique (limite de l'accepteur rapide, valable seulement s'il parcourt ce volume rapidement devant la durée de vie du donneur). Les deux diffèrent parce qu'un nuage est pondéré uniformément alors qu'une trajectoire l'est par les endroits où le fluorophore séjourne.

### `fret.dyes.1.D`

**Libellé:** Coefficient de diffusion de l'accepteur  
**Type:** `number`
**Unité:** `Å²/ns`

Coefficient de diffusion de l'accepteur FRET hors interaction avec la surface de la protéine.

### `fret.dyes.1.amino_acid_interactions`

**Libellé:** Interactions de l'accepteur  
**Type:** `table`

Table d'interaction par type de résidu pour l'accepteur, de même forme que l'amino_acid_interactions du donneur. Seul slow_factor est utilisé, car QuEst rapporte le déclin du donneur : l'extinction PET propre à l'accepteur modifierait sa brillance, pas celle du donneur. Par défaut, la table du donneur.

### `fret.dyes.1.attachment.atom`

**Libellé:** Atome de l'accepteur  
**Type:** `text`

Nom de l'atome sur lequel l'accepteur FRET est attaché.

### `fret.dyes.1.attachment.chain`

**Libellé:** Chaîne de l'accepteur  
**Type:** `text`

Identifiant de chaîne du résidu sur lequel l'accepteur FRET est attaché.

### `fret.dyes.1.attachment.residue`

**Libellé:** Résidu de l'accepteur  
**Type:** `integer`

Numéro de séquence du résidu sur lequel l'accepteur FRET est attaché.

### `fret.dyes.1.av_parameter.linker_length`

**Libellé:** Longueur du bras de l'accepteur  
**Type:** `number`
**Unité:** `Å`

Distance maximale entre le point d'attache et le centre de l'accepteur FRET.

### `fret.dyes.1.av_parameter.linker_width`

**Libellé:** Largeur du bras de l'accepteur  
**Type:** `number`
**Unité:** `Å`

Largeur du bras espaceur qui retient l'accepteur FRET.

### `fret.dyes.1.av_parameter.radius1`

**Libellé:** Rayon de l'accepteur  
**Type:** `number`
**Unité:** `Å`

Rayon d'exclusion représentant l'accepteur FRET comme une sphère.

### `fret.dyes.1.slow_radius`

**Libellé:** Rayon de ralentissement de l'accepteur  
**Type:** `number`
**Unité:** `Å`

Rayon autour de chaque centre d'acide aminé à l'intérieur duquel ce résidu ralentit l'accepteur. Distinct de celui du donneur : une cyanine et une rhodamine n'adhèrent pas de la même façon à une surface protéique. Par défaut, la valeur du donneur.

### `fret.enabled`

**Libellé:** FRET activé  
**Type:** `boolean`

Indique si un accepteur FRET est simulé afin de calculer l'efficacité de FRET et les courbes de déclin.

## Extinction

### `amino_acid_interactions`

**Libellé:** Interactions avec les acides aminés  
**Type:** `table`

Table d'interaction par type de résidu. Chaque entrée contient slow_factor (0–1, mesure du ralentissement du fluorophore par le résidu), kQ (taux d'extinction en 1/ns appliqué au contact), quench_radius (rayon de contact en ångströms depuis le centre du fluorophore ; null hérite de critical_distance) et quench_atoms (noms des atomes dont le barycentre définit le centre d'extinction). Par défaut, seuls les résidus à chaîne latérale rédox-active éteignent la fluorescence : TRP, TYR, MET, HIS, CYS et PRO.

### `amino_acid_interactions.kQ`

**Libellé:** Taux d'extinction kQ  
**Type:** `number`
**Unité:** `1/ns`

Taux auquel ce type de résidu éteint le fluorophore tant que les deux sont en contact. Les contributions des résidus dont les sphères de contact se recouvrent s'additionnent.

### `amino_acid_interactions.quench_atoms`

**Libellé:** Atomes du centre d'extinction  
**Type:** `list`

Noms des atomes dont le barycentre définit le centre d'extinction de ce résidu. Les valeurs par défaut suivent le groupement rédox-actif qui assure le transfert d'électron photo-induit : le cycle indole pour TRP, le cycle phénol pour TYR, le cycle imidazole pour HIS, SD pour MET et SG pour CYS — et non CB.

### `amino_acid_interactions.quench_radius`

**Libellé:** Rayon de contact  
**Type:** `number`
**Unité:** `Å`

Distance entre le centre du fluorophore et le centre d'extinction de ce résidu en deçà de laquelle l'extinction se produit. Laisser vide pour hériter du critical_distance du projet.

### `amino_acid_interactions.slow_factor`

**Libellé:** Facteur de ralentissement  
**Type:** `number`

Facteur compris entre 0 et 1 qui met à l'échelle le coefficient de diffusion du fluorophore au voisinage de ce type de résidu, modélisant une adhérence non spécifique. Les résidus qui se recouvrent multiplient leurs facteurs.

### `critical_distance`

**Libellé:** Rayon de contact par défaut  
**Type:** `number`
**Unité:** `Å`

Rayon de contact employé par les types de résidus qui ne définissent pas leur propre quench_radius. Il se mesure du centre du fluorophore au centre d'extinction du résidu et doit donc dépasser le rayon du fluorophore augmenté du contact de van der Waals (typiquement 7 à 9 Å pour un fluorophore de 3,5 Å).

## CLI override examples

- Set donor lifetime and attachment residue:
  ```bash
  quest simulate -p project.quest.json --set tau0=3.8 --set attachment.residue=55
  ```
- Sweep linker length and residue:
  ```bash
  quest simulate -p project.quest.json \
    --grid av_parameter.linker_length=10.0,12.5 \
    --grid attachment.residue=25,30,35
  ```

## Validation tips

- Ensure donor dye has `D` set; otherwise the API raises a `KeyError`.
- Provide a valid `pdb` path (or pass `--pdb`/`pdb_path`).
- When enabling FRET, include two dyes and an `R0_matrix` entry for donor→acceptor.
