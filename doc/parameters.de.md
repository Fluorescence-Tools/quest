# QuEst-Parameterreferenz

Diese Seite wird aus `parameter_catalog.json`, `quest/settings/locales/de.json` erzeugt. Ändern Sie zuerst dort den Hilfetext eines Parameters — Weboberfläche, diese Seite und das Desktop-Formular lesen dieselben Dateien.

## Übersicht des Katalogs

| Schlüssel | Kategorie | Typ | Einheit | Beschreibung |
| --- | --- | --- | --- | --- |
| `amino_acid_interactions` | Löschung | table | - | Tabelle der Wechselwirkungen je Resttyp. Jeder Eintrag enthält slow_factor (0–1, wie stark der Rest den diffundierenden Farbstoff verlangsamt), kQ (Löschrate in 1/ns bei Kontakt), quench_radius (Kontaktradius in Ångström vom Farbstoffmittelpunkt; null übernimmt critical_distance) und quench_atoms (Atomnamen, deren Schwerpunkt das Löschzentrum bildet). Standardmäßig löschen nur Reste mit redoxaktiver Seitenkette: TRP, TYR, MET, HIS, CYS und PRO. |
| `amino_acid_interactions.kQ` | Löschung | number | 1/ns | Rate, mit der dieser Resttyp den Farbstoff bei Kontakt löscht. Beiträge von Resten mit überlappenden Kontaktsphären addieren sich. |
| `amino_acid_interactions.quench_atoms` | Löschung | list | - | Atomnamen, deren Schwerpunkt das Löschzentrum dieses Rests bildet. Die Voreinstellungen folgen der redoxaktiven Gruppe, die den photoinduzierten Elektronentransfer vermittelt: der Indolring bei TRP, der Phenolring bei TYR, der Imidazolring bei HIS, SD bei MET und SG bei CYS – nicht CB. |
| `amino_acid_interactions.quench_radius` | Löschung | number | Å | Abstand vom Farbstoffmittelpunkt zum Löschzentrum dieses Rests, innerhalb dessen Löschung stattfindet. Leer lassen, um die projektweite critical_distance zu übernehmen. |
| `amino_acid_interactions.slow_factor` | Löschung | number | - | Faktor zwischen 0 und 1, der den Diffusionskoeffizienten des Farbstoffs in der Nähe dieses Resttyps skaliert und unspezifische Haftung beschreibt. Überlappende Reste multiplizieren ihre Faktoren. |
| `attachment.atom` | Markierungsstelle | text | - | Atomname (z. B. CB, CA), der als Anbindungspunkt des Farbstoffs dient. |
| `attachment.chain` | Markierungsstelle | text | - | Kettenbezeichner der Aminosäure, an die der Farbstoff gebunden ist. |
| `attachment.residue` | Markierungsstelle | integer | - | Sequenznummer der Aminosäure, an die der Farbstoff gebunden ist. |
| `av_parameter.linker_length` | Zugängliches Volumen | number | Å | Maximaler Abstand vom Anbindungspunkt zum Mittelpunkt des Farbstoffs. |
| `av_parameter.linker_width` | Zugängliches Volumen | number | Å | Breite des Linkers, der den Farbstoff hält. |
| `av_parameter.radius1` | Zugängliches Volumen | number | Å | Ausschlussradius, der den Farbstoff als Kugel beschreibt (Ein-Kugel-Näherung). |
| `coarse_grained` | Simulation | boolean | - | Reduziert die Struktur vor der Simulation auf das Rückgrat samt CB, wobei CB in den Schwerpunkt der Seitenkette verschoben wird. Seitenkettenatome fehlen dann, sodass das Löschzentrum jedes Rests auf dieses CB-Pseudoatom zurückfällt und das zugängliche Volumen wächst, weil die Seitenketten es nicht mehr versperren. Das ist eine Modellierungsentscheidung, keine Beschleunigung: Der Aufwand von QuEst wird vom AV-Gitter bestimmt, nicht von der Atomzahl. |
| `critical_distance` | Löschung | number | Å | Kontaktradius für Resttypen, die keinen eigenen quench_radius festlegen. Er wird vom Farbstoffmittelpunkt bis zum Löschzentrum des Rests gemessen und muss daher größer sein als der Farbstoffradius zuzüglich des Van-der-Waals-Kontakts (typischerweise 7–9 Å bei einem Farbstoff von 3,5 Å). |
| `dg` | Zugängliches Volumen | number | Å | Gitterauflösung für die Berechnung des zugänglichen Volumens (AV) und des Diffusionsraums. |
| `fret.R0_matrix` | FRET-Akzeptor | number | Å | Förster-Radius R0 des Donor-Akzeptor-Paares, in Ångström wie jede andere Länge im Projekt. Gespeichert wird die Zelle Donor→Akzeptor von fret.R0_matrix; die Transferrate skaliert mit (R0/r)^6, sodass eine Eingabe in Nanometern FRET verschwinden lässt. Typische Xanthen-Paare liegen bei 45–65 Å. |
| `fret.acceptor_dynamics` | FRET-Akzeptor | choice | - | Wie der Akzeptor behandelt wird: „trajectory“ simuliert seine Brownsche Bewegung mit eigenem Diffusionskoeffizienten und eigener Haftung und verwendet den momentanen Donor-Akzeptor-Abstand, „averaged“ mittelt über sein statisches zugängliches Volumen (Grenzfall des schnellen Akzeptors, nur gültig, wenn er dieses Volumen schnell im Vergleich zur Donorlebensdauer durchläuft). Beide unterscheiden sich, weil eine Punktwolke gleichmäßig gewichtet wird, eine Trajektorie hingegen danach, wo sich der Farbstoff tatsächlich aufhält. |
| `fret.dyes.0.D` | Simulation | number | Å²/ns | Diffusionskoeffizient des Donorfarbstoffs in Lösung, ohne Wechselwirkung mit der Proteinoberfläche. |
| `fret.dyes.1.D` | FRET-Akzeptor | number | Å²/ns | Diffusionskoeffizient des FRET-Akzeptorfarbstoffs ohne Wechselwirkung mit der Proteinoberfläche. |
| `fret.dyes.1.amino_acid_interactions` | FRET-Akzeptor | table | - | Tabelle der Wechselwirkungen je Resttyp für den Akzeptor, gleiche Form wie amino_acid_interactions des Donors. Verwendet wird nur slow_factor, denn QuEst berichtet das Abklingen des Donors; die eigene PET-Löschung des Akzeptors würde dessen Helligkeit ändern, nicht die des Donors. Voreinstellung ist die Tabelle des Donors. |
| `fret.dyes.1.attachment.atom` | FRET-Akzeptor | text | - | Atomname, an dem der FRET-Akzeptorfarbstoff gebunden ist. |
| `fret.dyes.1.attachment.chain` | FRET-Akzeptor | text | - | Kettenbezeichner des Rests, an den der FRET-Akzeptorfarbstoff gebunden ist. |
| `fret.dyes.1.attachment.residue` | FRET-Akzeptor | integer | - | Sequenznummer des Rests, an den der FRET-Akzeptorfarbstoff gebunden ist. |
| `fret.dyes.1.av_parameter.linker_length` | FRET-Akzeptor | number | Å | Maximaler Abstand vom Anbindungspunkt zum Mittelpunkt des FRET-Akzeptorfarbstoffs. |
| `fret.dyes.1.av_parameter.linker_width` | FRET-Akzeptor | number | Å | Breite des Linkers, der den FRET-Akzeptorfarbstoff hält. |
| `fret.dyes.1.av_parameter.radius1` | FRET-Akzeptor | number | Å | Ausschlussradius, der den FRET-Akzeptorfarbstoff als Kugel beschreibt. |
| `fret.dyes.1.slow_radius` | FRET-Akzeptor | number | Å | Radius um jedes Aminosäurezentrum, innerhalb dessen dieser Rest den Akzeptor verlangsamt. Getrennt vom Donor: Ein Cyanin und ein Rhodamin haften nicht in gleicher Weise an einer Proteinoberfläche. Voreinstellung ist der Wert des Donors. |
| `fret.enabled` | FRET-Akzeptor | boolean | - | Legt fest, ob ein FRET-Akzeptorfarbstoff simuliert wird, um FRET-Effizienz und Abklingkurven zu berechnen. |
| `fret.kappa2` | FRET-Akzeptor | number | - | Orientierungsfaktor κ² des Transfers, zwischen 0 (senkrecht zueinander stehende Übergangsdipole) und 4 (kollinear). QuEst modelliert ihn nicht: der Farbstoff ist ein Punkt, der in einem Volumen diffundiert, und besitzt keinen Dipol, der sich ausrichten könnte — dieser Wert ist also die Mittelungsannahme, die Sie treffen. Der Standardwert 2/3 ist das isotrope dynamische Mittel, das ein publizierter Förster-Radius bereits voraussetzt; ihn stehen zu lassen reproduziert die Rate exakt. Erhöhen Sie ihn nur, wenn belegt ist, dass beide Farbstoffe in ihrer Orientierung eingeschränkt sind. |
| `n_bins` | Simulation | integer | - | Anzahl der Zeitkanäle des ausgegebenen Abklinghistogramms. |
| `n_photons` | Simulation | integer | - | Anzahl der simulierten Anregungsphotonen zur Rekonstruktion des Fluoreszenzabklingens. |
| `output_file` | Erweitert | text | - | Namenspräfix für gespeicherte Simulationsergebnisse, Trajektorien und Abklingkurven. |
| `parallel_trajectories` | Erweitert | integer | - | Anzahl unabhängiger Brownscher Trajektorien, die gleichzeitig gerechnet und vor der Photonensimulation zusammengeführt werden. −1 nutzt alle CPU-Kerne, höchstens jedoch 8. |
| `pdb` | Struktur | file | - | Pfad zur Strukturdatei (PDB/CIF) oder eine vierstellige RCSB-PDB-Kennung; wird für die AV-Berechnung verwendet. |
| `random_seed` | Erweitert | integer | - | Startwert der Monte-Carlo-Simulation. Ist er gesetzt, werden sowohl die Diffusionstrajektorie als auch die Photonenspur initialisiert, sodass wiederholte Läufe desselben Projekts genau dasselbe Abklingen und dieselbe Quantenausbeute liefern. Ohne Startwert läuft die Simulation unabhängig zufällig; die Photonenspur nutzt dann einen schnelleren, mehrfädigen Kern. |
| `save_avs` | Erweitert | boolean | - | Legt fest, ob die berechneten Gitter des zugänglichen Volumens als XYZ-Dateien gespeichert werden. |
| `skip_frame` | Erweitert | integer | - | Unterabtastungsfaktor beim Schreiben von Trajektorienbildern in Koordinatendateien. |
| `slow_radius` | Erweitert | number | Å | Radius um jedes Aminosäurezentrum des gesamten Proteins, innerhalb dessen dieser Resttyp seinen slow_factor zum Gitter der Diffusionskoeffizienten beiträgt. |
| `t_max` | Simulation | number | ns | Gesamtlänge der simulierten Brownschen Trajektorie. |
| `t_step` | Simulation | number | ns | Schrittweite für die Berechnung der Brownschen Trajektorie. |
| `tau0` | Simulation | number | ns | Fluoreszenzlebensdauer des Donorfarbstoffs ohne Löschung. |

## Struktur

### `pdb`

**Bezeichnung:** Strukturdatei  
**Typ:** `file`

Pfad zur Strukturdatei (PDB/CIF) oder eine vierstellige RCSB-PDB-Kennung; wird für die AV-Berechnung verwendet.

## Markierungsstelle

### `attachment.atom`

**Bezeichnung:** Atom der Anbindung  
**Typ:** `text`

Atomname (z. B. CB, CA), der als Anbindungspunkt des Farbstoffs dient.

### `attachment.chain`

**Bezeichnung:** Kette der Anbindung  
**Typ:** `text`

Kettenbezeichner der Aminosäure, an die der Farbstoff gebunden ist.

### `attachment.residue`

**Bezeichnung:** Rest der Anbindung  
**Typ:** `integer`

Sequenznummer der Aminosäure, an die der Farbstoff gebunden ist.

## Zugängliches Volumen

### `av_parameter.linker_length`

**Bezeichnung:** Linkerlänge  
**Typ:** `number`
**Einheit:** `Å`

Maximaler Abstand vom Anbindungspunkt zum Mittelpunkt des Farbstoffs.

### `av_parameter.linker_width`

**Bezeichnung:** Linkerbreite  
**Typ:** `number`
**Einheit:** `Å`

Breite des Linkers, der den Farbstoff hält.

### `av_parameter.radius1`

**Bezeichnung:** Farbstoffradius  
**Typ:** `number`
**Einheit:** `Å`

Ausschlussradius, der den Farbstoff als Kugel beschreibt (Ein-Kugel-Näherung).

### `dg`

**Bezeichnung:** Gitterauflösung  
**Typ:** `number`
**Einheit:** `Å`

Gitterauflösung für die Berechnung des zugänglichen Volumens (AV) und des Diffusionsraums.

## Simulation

### `coarse_grained`

**Bezeichnung:** Grobkörnige Struktur  
**Typ:** `boolean`

Reduziert die Struktur vor der Simulation auf das Rückgrat samt CB, wobei CB in den Schwerpunkt der Seitenkette verschoben wird. Seitenkettenatome fehlen dann, sodass das Löschzentrum jedes Rests auf dieses CB-Pseudoatom zurückfällt und das zugängliche Volumen wächst, weil die Seitenketten es nicht mehr versperren. Das ist eine Modellierungsentscheidung, keine Beschleunigung: Der Aufwand von QuEst wird vom AV-Gitter bestimmt, nicht von der Atomzahl.

### `fret.dyes.0.D`

**Bezeichnung:** Diffusionskoeffizient des Donors  
**Typ:** `number`
**Einheit:** `Å²/ns`

Diffusionskoeffizient des Donorfarbstoffs in Lösung, ohne Wechselwirkung mit der Proteinoberfläche.

### `n_bins`

**Bezeichnung:** Kanäle des Abklingens  
**Typ:** `integer`

Anzahl der Zeitkanäle des ausgegebenen Abklinghistogramms.

### `n_photons`

**Bezeichnung:** Simulierte Photonen  
**Typ:** `integer`

Anzahl der simulierten Anregungsphotonen zur Rekonstruktion des Fluoreszenzabklingens.

### `t_max`

**Bezeichnung:** Simulationsdauer  
**Typ:** `number`
**Einheit:** `ns`

Gesamtlänge der simulierten Brownschen Trajektorie.

### `t_step`

**Bezeichnung:** Zeitschritt  
**Typ:** `number`
**Einheit:** `ns`

Schrittweite für die Berechnung der Brownschen Trajektorie.

### `tau0`

**Bezeichnung:** Ungelöschte Lebensdauer  
**Typ:** `number`
**Einheit:** `ns`

Fluoreszenzlebensdauer des Donorfarbstoffs ohne Löschung.

## Erweitert

### `output_file`

**Bezeichnung:** Präfix der Ausgabedateien  
**Typ:** `text`

Namenspräfix für gespeicherte Simulationsergebnisse, Trajektorien und Abklingkurven.

### `parallel_trajectories`

**Bezeichnung:** Parallele Trajektorien  
**Typ:** `integer`

Anzahl unabhängiger Brownscher Trajektorien, die gleichzeitig gerechnet und vor der Photonensimulation zusammengeführt werden. −1 nutzt alle CPU-Kerne, höchstens jedoch 8.

### `random_seed`

**Bezeichnung:** Zufallsstartwert  
**Typ:** `integer`

Startwert der Monte-Carlo-Simulation. Ist er gesetzt, werden sowohl die Diffusionstrajektorie als auch die Photonenspur initialisiert, sodass wiederholte Läufe desselben Projekts genau dasselbe Abklingen und dieselbe Quantenausbeute liefern. Ohne Startwert läuft die Simulation unabhängig zufällig; die Photonenspur nutzt dann einen schnelleren, mehrfädigen Kern.

### `save_avs`

**Bezeichnung:** AV-Dateien speichern  
**Typ:** `boolean`

Legt fest, ob die berechneten Gitter des zugänglichen Volumens als XYZ-Dateien gespeichert werden.

### `skip_frame`

**Bezeichnung:** Übersprungene Trajektorienbilder  
**Typ:** `integer`

Unterabtastungsfaktor beim Schreiben von Trajektorienbildern in Koordinatendateien.

### `slow_radius`

**Bezeichnung:** Verlangsamungsradius  
**Typ:** `number`
**Einheit:** `Å`

Radius um jedes Aminosäurezentrum des gesamten Proteins, innerhalb dessen dieser Resttyp seinen slow_factor zum Gitter der Diffusionskoeffizienten beiträgt.

## FRET-Akzeptor

### `fret.R0_matrix`

**Bezeichnung:** Förster-Radius  
**Typ:** `number`
**Einheit:** `Å`

Förster-Radius R0 des Donor-Akzeptor-Paares, in Ångström wie jede andere Länge im Projekt. Gespeichert wird die Zelle Donor→Akzeptor von fret.R0_matrix; die Transferrate skaliert mit (R0/r)^6, sodass eine Eingabe in Nanometern FRET verschwinden lässt. Typische Xanthen-Paare liegen bei 45–65 Å.

### `fret.acceptor_dynamics`

**Bezeichnung:** Akzeptordynamik  
**Typ:** `choice`

Wie der Akzeptor behandelt wird: „trajectory“ simuliert seine Brownsche Bewegung mit eigenem Diffusionskoeffizienten und eigener Haftung und verwendet den momentanen Donor-Akzeptor-Abstand, „averaged“ mittelt über sein statisches zugängliches Volumen (Grenzfall des schnellen Akzeptors, nur gültig, wenn er dieses Volumen schnell im Vergleich zur Donorlebensdauer durchläuft). Beide unterscheiden sich, weil eine Punktwolke gleichmäßig gewichtet wird, eine Trajektorie hingegen danach, wo sich der Farbstoff tatsächlich aufhält.

### `fret.dyes.1.D`

**Bezeichnung:** Diffusionskoeffizient des Akzeptors  
**Typ:** `number`
**Einheit:** `Å²/ns`

Diffusionskoeffizient des FRET-Akzeptorfarbstoffs ohne Wechselwirkung mit der Proteinoberfläche.

### `fret.dyes.1.amino_acid_interactions`

**Bezeichnung:** Wechselwirkungen des Akzeptors  
**Typ:** `table`

Tabelle der Wechselwirkungen je Resttyp für den Akzeptor, gleiche Form wie amino_acid_interactions des Donors. Verwendet wird nur slow_factor, denn QuEst berichtet das Abklingen des Donors; die eigene PET-Löschung des Akzeptors würde dessen Helligkeit ändern, nicht die des Donors. Voreinstellung ist die Tabelle des Donors.

### `fret.dyes.1.attachment.atom`

**Bezeichnung:** Atom des Akzeptors  
**Typ:** `text`

Atomname, an dem der FRET-Akzeptorfarbstoff gebunden ist.

### `fret.dyes.1.attachment.chain`

**Bezeichnung:** Kette des Akzeptors  
**Typ:** `text`

Kettenbezeichner des Rests, an den der FRET-Akzeptorfarbstoff gebunden ist.

### `fret.dyes.1.attachment.residue`

**Bezeichnung:** Rest des Akzeptors  
**Typ:** `integer`

Sequenznummer des Rests, an den der FRET-Akzeptorfarbstoff gebunden ist.

### `fret.dyes.1.av_parameter.linker_length`

**Bezeichnung:** Linkerlänge des Akzeptors  
**Typ:** `number`
**Einheit:** `Å`

Maximaler Abstand vom Anbindungspunkt zum Mittelpunkt des FRET-Akzeptorfarbstoffs.

### `fret.dyes.1.av_parameter.linker_width`

**Bezeichnung:** Linkerbreite des Akzeptors  
**Typ:** `number`
**Einheit:** `Å`

Breite des Linkers, der den FRET-Akzeptorfarbstoff hält.

### `fret.dyes.1.av_parameter.radius1`

**Bezeichnung:** Radius des Akzeptorfarbstoffs  
**Typ:** `number`
**Einheit:** `Å`

Ausschlussradius, der den FRET-Akzeptorfarbstoff als Kugel beschreibt.

### `fret.dyes.1.slow_radius`

**Bezeichnung:** Verlangsamungsradius des Akzeptors  
**Typ:** `number`
**Einheit:** `Å`

Radius um jedes Aminosäurezentrum, innerhalb dessen dieser Rest den Akzeptor verlangsamt. Getrennt vom Donor: Ein Cyanin und ein Rhodamin haften nicht in gleicher Weise an einer Proteinoberfläche. Voreinstellung ist der Wert des Donors.

### `fret.enabled`

**Bezeichnung:** FRET aktiviert  
**Typ:** `boolean`

Legt fest, ob ein FRET-Akzeptorfarbstoff simuliert wird, um FRET-Effizienz und Abklingkurven zu berechnen.

### `fret.kappa2`

**Bezeichnung:** Orientierungsfaktor κ²  
**Typ:** `number`

Orientierungsfaktor κ² des Transfers, zwischen 0 (senkrecht zueinander stehende Übergangsdipole) und 4 (kollinear). QuEst modelliert ihn nicht: der Farbstoff ist ein Punkt, der in einem Volumen diffundiert, und besitzt keinen Dipol, der sich ausrichten könnte — dieser Wert ist also die Mittelungsannahme, die Sie treffen. Der Standardwert 2/3 ist das isotrope dynamische Mittel, das ein publizierter Förster-Radius bereits voraussetzt; ihn stehen zu lassen reproduziert die Rate exakt. Erhöhen Sie ihn nur, wenn belegt ist, dass beide Farbstoffe in ihrer Orientierung eingeschränkt sind.

## Löschung

### `amino_acid_interactions`

**Bezeichnung:** Aminosäure-Wechselwirkungen  
**Typ:** `table`

Tabelle der Wechselwirkungen je Resttyp. Jeder Eintrag enthält slow_factor (0–1, wie stark der Rest den diffundierenden Farbstoff verlangsamt), kQ (Löschrate in 1/ns bei Kontakt), quench_radius (Kontaktradius in Ångström vom Farbstoffmittelpunkt; null übernimmt critical_distance) und quench_atoms (Atomnamen, deren Schwerpunkt das Löschzentrum bildet). Standardmäßig löschen nur Reste mit redoxaktiver Seitenkette: TRP, TYR, MET, HIS, CYS und PRO.

### `amino_acid_interactions.kQ`

**Bezeichnung:** Löschrate kQ  
**Typ:** `number`
**Einheit:** `1/ns`

Rate, mit der dieser Resttyp den Farbstoff bei Kontakt löscht. Beiträge von Resten mit überlappenden Kontaktsphären addieren sich.

### `amino_acid_interactions.quench_atoms`

**Bezeichnung:** Atome des Löschzentrums  
**Typ:** `list`

Atomnamen, deren Schwerpunkt das Löschzentrum dieses Rests bildet. Die Voreinstellungen folgen der redoxaktiven Gruppe, die den photoinduzierten Elektronentransfer vermittelt: der Indolring bei TRP, der Phenolring bei TYR, der Imidazolring bei HIS, SD bei MET und SG bei CYS – nicht CB.

### `amino_acid_interactions.quench_radius`

**Bezeichnung:** Kontaktradius  
**Typ:** `number`
**Einheit:** `Å`

Abstand vom Farbstoffmittelpunkt zum Löschzentrum dieses Rests, innerhalb dessen Löschung stattfindet. Leer lassen, um die projektweite critical_distance zu übernehmen.

### `amino_acid_interactions.slow_factor`

**Bezeichnung:** Verlangsamungsfaktor  
**Typ:** `number`

Faktor zwischen 0 und 1, der den Diffusionskoeffizienten des Farbstoffs in der Nähe dieses Resttyps skaliert und unspezifische Haftung beschreibt. Überlappende Reste multiplizieren ihre Faktoren.

### `critical_distance`

**Bezeichnung:** Voreingestellter Kontaktradius  
**Typ:** `number`
**Einheit:** `Å`

Kontaktradius für Resttypen, die keinen eigenen quench_radius festlegen. Er wird vom Farbstoffmittelpunkt bis zum Löschzentrum des Rests gemessen und muss daher größer sein als der Farbstoffradius zuzüglich des Van-der-Waals-Kontakts (typischerweise 7–9 Å bei einem Farbstoff von 3,5 Å).

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
