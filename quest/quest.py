#!/usr/bin/python
from __future__ import print_function
from lib.structure import Structure
import itertools
import argparse
import numpy as np
import json
import quest_gui

parser = argparse.ArgumentParser(description='Queching estimation by PDB file.')
parser.add_argument('-gui', '--gui', action='store_true', dest='gui')
parser.add_argument('-f', metavar='pdb_file', type=str, help='The pdb file used', required=False)
parser.add_argument('-c', metavar='chain_id', type=str, help='The chain id.', required=False)
parser.add_argument('-p', metavar='labeling_positions', type=int, help='The residue numbers of the amino acids to be labeled.', nargs='+', required=False)
parser.add_argument('-o', metavar='qy_output_file', type=str, help='The file to which the QY is written.', default="QY_out.txt")
parser.add_argument('-d', metavar='diffusion_coefficient', type=float, default=7.5, help='The diffusion coefficient in Ang2/ns')
parser.add_argument('-s', metavar='save_decay', type=bool, help='Save the fluorescence decays to a file.', default=False)
parser.add_argument('-a', metavar='attachment_atom', type=str, help='The attachment atom of the label.', default='CB')
parser.add_argument('-u', metavar='slow_radius', type=float, help='The radius around the C-beta atoms which is considered as the slow region of the accessible volume.', default=8.5)
parser.add_argument('-t', metavar='slow_fact', type=float, help='The factor by which the diffusion coefficient of the dyes is multiplied in case they are in a slow region of the accessible volume.', default=0.1)
parser.add_argument('-e', metavar='critical_distance', type=float, help='The distance below the dye is quenched by a quencher.', default=8.5)
parser.add_argument('-n', metavar='n_photons', type=int, help='The number of generated photons.', default=5000000)
parser.add_argument('-l', metavar='linker_length', type=float, help='The linker length of the dye.', default=22.0)
parser.add_argument('-r', metavar='dye_radius', type=float, help='The radius of the dye.', default=3.0)
parser.add_argument('-w', metavar='linker_width', type=float, help='The linker width of the dye.', default=3.0)
parser.add_argument('-m', metavar='simulation_time', type=float, help='Simulation time given in nanoseconds.', default=16000)
parser.add_argument('-st', metavar='simulation_time_step', type=float, help='Simulation time-step given in nanoseconds.', default=0.032)
parser.add_argument('-lt', metavar='fluorescence_lifetime', type=float, help='Fluorescence lifetime of the unquenched dye.', default=4.2)
parser.add_argument('-tacrange', metavar='tac_range', type=float, help='TAC range of the decay histogram.', nargs='+', default=(0, 100))
parser.add_argument('-tac_dt', metavar='tac_range', type=float, help='Resoltion of the TAC histogram.', default=0.128)
parser.add_argument('-rep', metavar='repetitions', type=int, help='Number of repetitions for each simulations.', default=3)
parser.add_argument('-find_idx', metavar='find_atom_idx', type=int, help='If set to zero only use atom index specified in JSON file.', default=1)
parser.add_argument('-json', metavar='json_file', type=str, help='Find the index of all quenching atoms. Otherwise only use atom index specified in JSON file.', default='quencher.json')
parser.add_argument('-avnp', metavar='av_npoints', type=int, help='Threshold below which an AV is discarded relative value of number of points in the cube.', default=1000)
args = parser.parse_args()

if args.gui:
    quest_gui.start_gui()
else:
    pass
    # print("")
    # print("============================")
    # print("Quenching Estimation - QuEst")
    # print("============================")
    # print("PDB-File: \t\t%s" % args.f)
    # print("Diff. coeff. [A2/ns] :\t%s" % args.d)
    # print("Labling positions :\t%s" % args.p)
    # print("Chain ID:\t%s" % args.c)
    # print("Attachment_atom: \t%s" % args.a)
    # print("Save decys: \t\t%s" % args.s)
    # print("Critical distance [A]: \t%s" % args.e)
    # print("Number of photons: \t%s" % args.n)
    # print("Slow_radius [A]: \t%s" % args.u)
    # print("Dye linker lenght [A]: \t%s" % args.l)
    # print("Dye linker width [A]: \t%s" % args.w)
    # print("Dye radius [A]: \t%s" % args.r)
    # print("Sim. time [ns]: \t%s" % args.m)
    # print("Sim. time step [ns]: \t%s" % args.st)
    # print("Flu. lifetime [ns]: \t%s" % args.lt)
    # print("TAC range [ns]: \t" + str(args.tacrange))
    # print("TAC dt [ns]: \t%s" % args.tac_dt)
    # print("Output file: \t%s" % args.o)
    # print("Repetitions: \t%s" % args.rep)
    # print("JSON file: \t%s" % args.json)
    # print("============================")
    # print("")
    #
    # pdb_file = args.f
    # diffusion_coefficient = args.d
    # labeling_positions = args.p
    # chain_id = args.c
    # #pdb_file = '../sample_data/model/hgbp1/hGBP1_coarse_all_ala.pdb'
    # structure = Structure(pdb_file)
    # directory = "./tmp/"                  # This directory is used for saving
    # slow_radius = args.u
    # slow_fact = args.t
    # attachment_atom = args.a
    # save_decays = args.s
    # critical_distance = args.e #3.0 + 5.5
    # n_photons = args.n
    # av_length = args.l
    # av_radius = args.r
    # av_width = args.w # check i think this is not passed
    # t_max = args.m
    # t_step = args.st
    # tau0 = args.lt
    # tac_range = args.tacrange
    # dt_tac = args.tac_dt
    #
    # donor_chains = [chain_id]*len(labeling_positions)
    # quencher = json.load(open(args.json))
    # donor_quenching = ProteinQuenching(structure, all_atoms_quench=False, quench_scale=1.0, quencher=quencher)
    #
    # donor_sticking = Sticking(structure, donor_quenching,
    #                           sticky_mode='surface',
    #                           slow_radius=slow_radius,
    #                           slow_fact=slow_fact)  # Stas-paper Dye-MD (roughly 10%)
    #
    # donor_dyes = dict(
    #     [
    #         (pos,
    #          Dye(donor_sticking,
    #              attachment_residue=pos,
    #              attachment_chain=chain,
    #              attachment_atom=attachment_atom,
    #              critical_distance=critical_distance,  # 3.0 Ang von AV + 6.0 Ang only C-beta quench (so far best 5.0 + 3.0)
    #              diffusion_coefficient=diffusion_coefficient,  # Stas-paper (Dye-MD 30 A2/ns)
    #              av_radius=av_radius,
    #              av_length=av_length,  # 20 + 5 = 3.5 + 21.5,
    #              av_width=av_width,
    #              tau0=tau0)
    #         )
    #         for pos, chain in zip(labeling_positions, donor_chains)
    #     ]
    # )
    #
    # simulation_parameter = DiffusionSimulationParameter(t_max=t_max,
    #                                                     t_step=t_step)
    # decay_parameter = DecaySimulationParameter(decay_mode='photon',
    #                                            n_photons=n_photons,
    #                                            tac_range=tac_range,
    #                                            dt_tac=dt_tac)
    #
    #
    #
    # def simulate_decays(dyes, decay_parameter, simulation_parameter, quenching_parameter, save_decays=True,
    #                     directory="./"):
    #     dye_decays = dict()
    #     print("Simulating labeling position: ", end="")
    #     with open(args.o, 'w') as fp:
    #         fp.write("LP\tQY\n")
    #     for dye_key in dyes:
    #         print( "%s " % dye_key, end="")
    #         with open(args.o, 'a') as fp:
    #             fp.write("%s\t" % (dye_key))
    #         for r in range(args.rep):
    #             dye = dyes[dye_key]
    #             av = dye.get_av()
    #             n_points = av.points.shape[0]
    #             if n_points > args.avnp:
    #                 diffusion_simulation = DiffusionSimulation(dye,
    #                                                            quenching_parameter,
    #                                                            simulation_parameter)
    #                 diffusion_simulation.update()
    #                 av = diffusion_simulation.av
    #                 #av.save('%sD' % dye_key)
    #                 #diffusion_simulation.save('%sD_diffusion.xyz' % dye_key, mode='xyz', skip=5)
    #                 fd0_sim_curve = DyeDecay(decay_parameter, diffusion_simulation)
    #                 fd0_sim_curve.update()
    #                 decay = fd0_sim_curve.get_histogram()
    #                 with open(args.o, 'a') as fp:
    #                     fp.write("\t%.2f\t" % (fd0_sim_curve.quantum_yield))
    #                 decay = np.vstack(decay)
    #             else:
    #                 with open(args.o, 'a') as fp:
    #                     fp.write("NA\t")
    #         with open(args.o, 'a') as fp:
    #             fp.write("\n")
    #
    # simulate_decays(donor_dyes, decay_parameter, simulation_parameter, donor_quenching, save_decays=save_decays)
    #
    # print("\nSimulation finished!\n")
