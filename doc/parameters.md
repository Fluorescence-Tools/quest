# Parameters

## Settings file


## Sticking

| Parameter name | Abbreviation | Description |
|--------------------------------|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Definition of sticky surface | Surface / Quencher | The radio button defines which part of the protein is "sticky". If the option Surface is chosen, the entire protein surface is sticky, meaning slows down the diffusion of the dye. If quencher is selected, only quenching amino acids slow down the dye's diffusion. |
| Dye surface interaction radius | Rs [A] | Dyes which are closer by a distance R to an atom than the distance "Rs" diffuse slower by the factor "slow factor". Here, the distance R is the distance between the center of the dye and the center of the atom. |
| Factor by which an dye close to the surface is slowed down | slow factor | The diffusion coefficient of the dye is multiplied by this factor if the dye is in vicinity of the protein's surface |

## Label

| Parameter name | Abbreviation | Description |
|------------------------------------------|--------------|-----------------------------------------------------------------------------------------------------------------|
| Chain ID to which the dye is attached to | Chain | The attachment position of the dye linker is selected first by the chain ID |
| The resiude ID of attachment | Residue | The residue ID defines to which residue the dye is attached to |
| The name of the atom of attachment | Atom | The atom name within a residue selects the atom to which the dye is attached to. |
| Linker length of the dye | Length | The linker length is the maximum distance from the attachment point of the dye to the center of the fluorophore |
| Linker width of the dye linker | Width | The width of the linker connecting the dye to the attachment atom position |
| Dye radius | Radius | The dye is approximated by a single sphere with a radius specified by this parameter |
| Simulation grid resolution | dg | The dye diffuses on a grid with a resolution defined by this parameter. |

## Dye

| Parameter name                                                            | Abbreviation | Description                                                                                        |
|---------------------------------------------------------------------------|--------------|----------------------------------------------------------------------------------------------------|
| Fluorescence lifetime of the dye in the absence of quenching              | tau0         | This parameter specifies the fluorescence lifetime of the dye in the absence of dynamic quenching. |
| Diffusion coefficient of the dye that is not interacting with the surface | D [A2/ns]    | The diffusion coefficient of the dye that is not in proximity of the protein surface.              |

## Quencher

| Parameter name                                          | Abbreviation   | Description                                                                                                                                                                                            |
|---------------------------------------------------------|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| List of quenching amino acids                           | Quencher       | This parameter defines which amino acids are quenching the dye. The amino acids are identified by their 3-letter code. Multiple amino-acids can be specified by separating 3-letter codes with spaces. |
| Distance below which a dye is quenched by an amino acid | Quench radius  | A dye that is closed than "Quench" radius to a quenching amino acid is quenched. The distance is the distance between the center of the dye and an atom of the quenching amino acid.                   |
| Quenching rate constant of the quenching amino acid     | kQ [1/ns]      | This parameter specifies the quenching rate constant of the quenching amino acids. Dyes that are closer than "Quench radius" to a quenching amino acid are quenched with the specified rate constant.  |
| List of atom names that are not quenching the dye       | Excluded atoms | If the checkbox excluded atoms is checked the atom names specified in the text box below are not considered to be quenching the dye's fluorescence.                                                    |

## Simulation

| Parameter name                                   | Abbreviation       | Description                                                                                           |
|--------------------------------------------------|--------------------|-------------------------------------------------------------------------------------------------------|
| Simulation time                                  | sim time [µs]      | The length of the Brownian dynamics (BD) simulation                                                   |
| Simulation time step                             | dt [ps]            | The step size of the BD simulation                                                                    |
| Number of simulated photons                      | nPhotons [Million] | The number of photons that are generated                                                              |
| Number of simulation frames                      | frames             | The total number of frames (this parameter is calculated using the simulation time and the time-step) |
| Show the entire AV in the 3D illustration window | show AV            | If this checkbox is checked, the AV is shown in the 3D-Illustration window.                           |


## Results

| Parameter name                                                     | Abbreviation | Description                                                                                                                                                                                                                                               |
|--------------------------------------------------------------------|--------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Simulated fluorescence quantum yield                               | QY(F)        | The simulated fluorescence quantum yield of the bright species.                                                                                                                                                                                           |
| Fraction of frames where a dye was close to a quenching amino acid | collided [%] | This number corresponds to the fraction of frames, where a dye was closer to a quenching amino acid than the distance "Quench radius".                                                                                                                    |
| Number of bins for in the simulated fluorescence decay             | nBins        | The total number of bins of the simulated fluorescence decay                                                                                                                                                                                              |
| Range of the fluorescence decay                                    | range        |                                                                                                                                                                                                                                                           |
| Skip parameter for 3D illustration                                 | skip         | In the 3D illustration only fraction of the frames are displayed. The integer number specifies how many frames are display, e.g., a value of 40 means that only every 40th frames in displayed in the 3D window. For the calculation all frames are used. |

