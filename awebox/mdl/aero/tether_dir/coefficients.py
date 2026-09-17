#
#    This file is part of awebox.
#
#    awebox -- A modeling and optimization framework for multi-kite AWE systems.
#    Copyright (C) 2017-2020 Jochem De Schutter, Rachel Leuthold, Moritz Diehl,
#                            ALU Freiburg.
#    Copyright (C) 2018-2020 Thilo Bronnenmeyer, Kiteswarms Ltd.
#    Copyright (C) 2016      Elena Malz, Sebastien Gros, Chalmers UT.
#
#    awebox is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    awebox is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with awebox; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
'''
file to generate the tether drag coefficient vs. reynolds number relationship, for a long cylinder
_python-3.5 / casadi-3.4.5
- author: rachel leuthold, alu-fr 2018
'''

import os
import matplotlib
if "MPLBACKEND" not in os.environ:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import casadi.tools as cas
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.tools.struct_operations as struct_op
import awebox.tools.vector_operations as vect_op

def get_tether_cd_fun(model_options, parameters, cd0_overwrite=None, cd_model_overwrite=None, smoothing_overwrite=None):
    info_to_add_to_applied_params_dict = {}
    reynolds = cas.SX.sym('reynolds')

    if cd_model_overwrite is not None:
        cd_model = cd_model_overwrite
    else:
        cd_model = model_options['tether']['cd_model']

    if smoothing_overwrite is not None:
        smoothing = smoothing_overwrite
    else:
        smoothing = model_options['tether']['reynolds_smoothing']

    if cd0_overwrite is not None:
        cd0 = cd0_overwrite
    else:
        cd0 = parameters['theta0','tether','cd']

    info_to_add_to_applied_params_dict['model.tether.cd_model'] = cd_model
    if cd_model == 'polyfit':
        drag_coeff = get_interpolation(reynolds, smoothing)
        info_to_add_to_applied_params_dict['model.tether.reynolds_smoothing'] = smoothing
    elif cd_model == 'piecewise':
        drag_coeff = get_roshko_unitstep(reynolds, smoothing)
        info_to_add_to_applied_params_dict['model.tether.reynolds_smoothing'] = smoothing
    elif cd_model == 'constant':
        drag_coeff = cd0
        info_to_add_to_applied_params_dict['params.tether.cd'] = drag_coeff
    else:
        raise ValueError('invalid tether drag coefficient model selected: %s', cd_model)

    tether_cd_fun = cas.Function('tether_cd_fun', [reynolds, parameters], [drag_coeff])

    return tether_cd_fun, info_to_add_to_applied_params_dict

def test(thresh=1.2):
    # thresh 1.0 means maximally finding value twice expected.
    test_dict = {}

    heddleson_re, heddleson_cd = get_heddleson_datapoints()
    for idx in range(len(heddleson_re)):
        test_dict['Heddleson' + str(idx)] = {'reynolds': heddleson_re[idx], 'cd': heddleson_cd[idx]}

    achenbach_re, achenbach_cd = get_achenbach_datapoints()
    for idx in range(len(achenbach_re)):
        test_dict['Achenbach' + str(idx)] = {'reynolds': achenbach_re[idx], 'cd': achenbach_cd[idx]}

    re_sym = cas.SX.sym('re_sym')
    smoothing = 1e-15
    cd0 = 1.
    for cd_model_overwrite in ['polyfit', 'piecewise']:

        options = {'model': {'tether': {'cd_model': cd_model_overwrite, 'reynolds_smoothing': smoothing}},
                   'params': {'tether': {'cd': cd0}}}
        parameters_dict = {}
        parametric_options = options['params']
        parameters_dict['theta0'] = struct_op.generate_nested_dict_struct(parametric_options)
        parameters = cas.struct_symSX([
            cas.entry('theta0', struct=parameters_dict['theta0'])
        ])

        cd_fun, _ =  get_tether_cd_fun(options['model'], parameters, cd0_overwrite=cd0, cd_model_overwrite=cd_model_overwrite, smoothing_overwrite=smoothing)
        cd_temp = cd_fun(re_sym, parameters(cd0))
        local_fun = cas.Function('local_fun', [re_sym], [cd_temp])

        for test_name, test_vals in test_dict.items():
            found = local_fun(test_vals['reynolds'])
            test_dict[test_name]['cd_found'] = found

            expected = test_vals['cd']
            error = (expected - found) / expected
            test_dict[test_name]['error'] = error
            criteria = (error * error)**0.5 < thresh
            if not criteria:
                message = 'unexpected cd found with ' + cd_model_overwrite + ' model at test ' + test_name + '. '
                for val_name, val_val in test_dict[test_name].items():
                    message += val_name + ': ' + str(val_val) + ', '
                message = message[:-2]
                print_op.log_and_raise_error(message)
    return None

def plot_cd_vs_reynolds(num_fig, model_options=None, smoothing=1e-15, cd=1):

    log_reynolds_list = np.linspace(0., 7., 100)
    reynolds_list = vect_op.columnize(10.**cas.DM(log_reynolds_list))

    re_sym = cas.SX.sym('re_sym')

    cd_list_dict = {}
    for cd_model_overwrite in ['constant', 'polyfit', 'piecewise']:

        options = {'model': {'tether': {'cd_model': cd_model_overwrite, 'reynolds_smoothing': smoothing}},
                   'params': {'tether': {'cd': cd}}}
        parameters_dict = {}
        parametric_options = options['params']
        parameters_dict['theta0'] = struct_op.generate_nested_dict_struct(parametric_options)
        parameters = cas.struct_symSX([
            cas.entry('theta0', struct=parameters_dict['theta0'])
        ])

        cd_fun, _ =  get_tether_cd_fun(model_options, parameters, cd0_overwrite=cd, cd_model_overwrite=cd_model_overwrite, smoothing_overwrite=smoothing)
        cd_temp = cd_fun(re_sym, parameters(cd))
        local_fun = cas.Function('local_fun', [re_sym], [cd_temp])
        drag_map = local_fun.map(len(log_reynolds_list), 'serial')
        cd_list_dict[cd_model_overwrite] = vect_op.columnize(drag_map(reynolds_list))
    for name, val in cd_list_dict.items():
        cd_list_dict[name] = np.array(val)

    reynolds_list = np.array(reynolds_list)

    heddleson_re, heddleson_cd = get_heddleson_datapoints()
    achenbach_re, achenbach_cd = get_achenbach_datapoints()
    yuce_re, yuce_cd = get_yuce_datapoints()
    hoerner_re, hoerner_cd = get_hoerner_datapoints()

    plt.figure(num_fig)
    for name, val in cd_list_dict.items():
        plt.loglog(reynolds_list, val, label=name)
    plt.loglog(achenbach_re, achenbach_cd, 'r*', label='Achenbach (1968)')
    plt.loglog(heddleson_re, heddleson_cd, 'b*', label='Heddleson et al. (1957)')
    plt.loglog(hoerner_re, hoerner_cd, 'm*', label='Hoerner (1965)')
    # plt.loglog(yuce_re, yuce_cd, 'y*', label='Yuce & Kareem (2016)')

    plt.ylim(1e-2, 1e2)
    plt.xlim(1., reynolds_list[-1])

    cd_piecewise_list = get_cd_piecewise_info_list(re_sym)
    cd_height = plt.ylim()[0]
    for idx in range(len(cd_piecewise_list)):
        cdp_dict = cd_piecewise_list[idx]
        if idx == 0:
            plt.plot([cdp_dict['re_start'], cdp_dict['re_end']], 2 * [cd_height], 'k+-', label='Zdravkovich (1990) regimes')
        else:
            plt.plot([cdp_dict['re_start'], cdp_dict['re_end']], 2 * [cd_height], 'k+-')
        plt.plot(2 * [cdp_dict['re_start']], plt.ylim(), 'k--', linewidth=0.1)

        if (cdp_dict['re_start'] < plt.xlim()[-1]) and (cdp_dict['re_end'] > plt.xlim()[0]):
            re_min = np.max(np.array([plt.xlim()[0], cdp_dict['re_start']]))
            re_max = np.min(np.array([plt.xlim()[-1], cdp_dict['re_end']]))
            plt.text(x=(re_min * re_max)**0.5, y= 1.2 * cd_height, s= cdp_dict['regime'], ha='center', fontsize=5, rotation=90)

    plt.xlabel('Reynolds number [-]')
    plt.ylabel('drag coefficient [-]')
    plt.title('comparison of drag coefficient models available in the awebox')
    plt.xlim(reynolds_list[0], reynolds_list[-1])
    plt.legend(loc='upper right')
    plt.show()

def get_cd_piecewise_info_list(re_sym, cd_outside_bounds=1e3):
    cd_piecewise_list = []

    def inv(xsym, x0, x1, y0, y1):
        # y = m/x + b
        # y1 = m / x1 + b
        # y0 = m / x0 + b
        m = (y1 - y0) * x0 * x1 / (x0 - x1)
        b = y0 - m / x0
        eq = m / xsym + b
        x_match = m / (cd_outside_bounds - b)
        return eq, x_match

    def linear(xsym, x0, x1, y0, y1):
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        return xsym * m + b

    def semiloglinear(xsym_log, x0, x1, y0, y1):
        return linear(cas.log10(xsym_log), cas.log10(x0), cas.log10(x1), y0, y1)

    def logloglinear(xsym_log, x0, x1, y0, y1):
        exp = linear(cas.log10(xsym_log), cas.log10(x0), cas.log10(x1), cas.log10(y0), cas.log10(y1))
        return 10.**exp

    def parabolic(xsym, x0, x1, x2, y0, y1, y2):
        p0 = (xsym - x1) * (xsym - x2) / (x0 - x1) / (x0 - x2)
        p1 = (xsym - x0) * (xsym - x2) / (x1 - x0) / (x1 - x2)
        p2 = (xsym - x0) * (xsym - x1) / (x2 - x0) / (x2 - x1)
        return y0 * p0 + y1 * p1 + y2 * p2

    def semilogpara(xsym, x0, x1, x2, y0, y1, y2):
        return parabolic(cas.log10(xsym), cas.log10(x0), cas.log10(x1), cas.log10(x2), y0, y1, y2)

    # State and region names from
    # https://flore.unifi.it/retrieve/e398c378-e5f2-179a-e053-3705fe0a4cff/LUPI_Tesi%20Dottorato_2.pdf
    # reproducing images from (Zdravkovich, 1997)
    # page 4 (tab 3.1), page 12 (fig 3.3)

    laminar_eq, laminar_match = inv(re_sym, 1e0, 1e3, 1.2e1, 1.)
    Lamnr = {'state': 'laminar',
             'regime': 'L1: no-separation - L3: periodic wake',
             'cd': laminar_eq,
             're_start': laminar_match,
             're_end': 1e2,
             'source': 'Cylinder Drag, scienceworld.wolfram.com'} #https://scienceworld.wolfram.com/physics/CylinderDrag.html
    cd_piecewise_list += [Lamnr]

    TrWke = {'state': 'transition in wake',
             'regime': 'TrW1: lower - TrW2: upper transition',
             'cd': laminar_eq,
             're_start': 1e2,
             're_end': 4e2,
             'source': 'Hoerner (1965)'} #https://scienceworld.wolfram.com/physics/CylinderDrag.html
    cd_piecewise_list += [TrWke]

    TrSL1 = {'state': 'transition in shear layers',
            'regime': 'TrSL1: lower subcritical',
             'cd': laminar_eq,
             're_start': 4e2,
             're_end': 1e3,
             'source': 'Hoerner (1965)'} #https://scienceworld.wolfram.com/physics/CylinderDrag.html
    cd_piecewise_list += [TrSL1]

    TrSL2 = {'state': 'transition in shear layers',
            'regime': 'TrSL2: intermediate subcritical',
             'cd': 1, #1.,
             're_start': 1e3,
             're_end': 6e3,
             'source': 'Cylinder Drag, scienceworld.wolfram.com'} #https://scienceworld.wolfram.com/physics/CylinderDrag.html
    cd_piecewise_list += [TrSL2]

    TrSL3 = {'state': 'transition in shear layers',
            'regime': 'TrSL3: upper subcritical',
             'cd': semilogpara(re_sym, 6e3, (6e3 * 2e5)**0.5, 2e5, 1., 1.22, 1.),
             're_start': 6e3,
             're_end': 2e5,
             'source': 'Hoerner (1965)'} #https://digital.library.unt.edu/ark:/67531/metadc56716/m2/1/high_res_d/19930083675.pdf
    cd_piecewise_list += [TrSL3]

    TrBL0 = {'state': 'transition in boundary layers',
            'regime': 'TrBL0: pre-critical',
             'cd': logloglinear(re_sym, 2e5, 3e5, 1., 0.9),
             're_start': 2e5,
             're_end': 3e5,
             'source': 'Delany & Sorensen (1953)'} #https://digital.library.unt.edu/ark:/67531/metadc56716/m2/1/high_res_d/19930083675.pdf
    cd_piecewise_list += [TrBL0]

    TrBL1 = {'state': 'transition in boundary layers',
            'regime': 'TrBL1: single bubble',
             'cd': logloglinear(re_sym, 3e5, 4e5, 0.9, 0.22),
             're_start': 3e5,
             're_end': 4e5,
             'source': 'Delany & Sorensen (1953)'
             }
    cd_piecewise_list += [TrBL1]

    TrBL2 = {'state': 'transition in boundary layers',
             'regime': 'TrBL2: two-bubble',
             'cd': logloglinear(re_sym, 4e5, 1e6, 0.22, 0.3),
             're_start': 4e5,
             're_end': 1e6,
             'source': 'Delany & Sorensen (1953)'} #https://digital.library.unt.edu/ark:/67531/metadc56716/m2/1/high_res_d/19930083675.pdf
    cd_piecewise_list += [TrBL2]

    TrBL3 = {'state': 'transition in boundary layers',
             'regime': 'TrBL3: supercritical',
             'cd': semiloglinear(re_sym, 1e6, 3.5e6, 0.3, 0.7),
             're_start': 1e6,
             're_end': 3.5e6,
             'source': 'Roshko (1961)'} #https://authors.library.caltech.edu/records/m8vtc-33e74
    cd_piecewise_list += [TrBL3]

    TrBL4 = {'state': 'transition in boundary layers',
             'regime': 'TrBL4: post-critical',
             'cd': 0.7,
             're_start':3.5e6,
             're_end': 1e7,
             'source': 'Roshko (1961)'} #https://authors.library.caltech.edu/records/m8vtc-33e74
    cd_piecewise_list += [TrBL4]

    Turbt = {'state': 'fully turbulent',
             'regime': 'T1: invariable - T2: ultimate',
             'cd': 0.7,
             're_start': 1e7,
             're_end': 1e12,
             'source': 'Zdravkovich (1990)'} #https://flore.unifi.it/retrieve/e398c378-e5f2-179a-e053-3705fe0a4cff/LUPI_Tesi%20Dottorato_2.pdf
    cd_piecewise_list += [Turbt]

    for idx in range(len(cd_piecewise_list)):
        cdp_dict = cd_piecewise_list[idx]
        loc_name = cdp_dict['regime'].replace(' ', '').replace(':', '').replace('-', "_")
        if isinstance(cdp_dict['cd'], float):
            cdp_dict['cd'] = cas.DM(cdp_dict['cd'])
        cdp_dict['cd_fun'] = cas.Function('cd_fun', [re_sym], [cdp_dict['cd']])

    # import pdb; pdb.set_trace()

    return cd_piecewise_list

def get_roshko_unitstep(reynolds, eps=1e-4):

    re_sym = cas.SX.sym('re_sym')
    log_re_sym = cas.log10(re_sym)

    cd_outside_bounds = 1e3
    cd_piecewise_list = get_cd_piecewise_info_list(re_sym, cd_outside_bounds=cd_outside_bounds)

    interpted = cd_outside_bounds * (1. - vect_op.unitstep(log_re_sym - cas.log10(cd_piecewise_list[0]['re_start']), eps))
    for idx in range(len(cd_piecewise_list)):
        cdp_dict = cd_piecewise_list[idx]
        interpted += vect_op.step_in_out(log_re_sym, cas.log10(cdp_dict['re_start']), cas.log10(cdp_dict['re_end']), eps) * cdp_dict['cd']
    interpted += cd_outside_bounds * vect_op.unitstep(log_re_sym - cas.log10(cd_piecewise_list[-1]['re_end']), eps)

    interp_fun = cas.Function('roshko_fun', [re_sym], [interpted])
    return interp_fun(reynolds)


def get_heddleson_datapoints():
    # Heddleson et al, 1957 https://apps.dtic.mil/sti/tr/pdf/ADA395503.pdf
    test_dict = {'Heddleson0': {'reynolds': 4e4, 'cd': 1e0},
                 'Heddleson1': {'reynolds': 1e5, 'cd': 1e0},
                 'Heddleson2': {'reynolds': 4e5, 'cd': 5e-1},
                 'Heddleson3': {'reynolds': 7e5, 'cd': 3e-1},
                 'Heddleson4': {'reynolds': 2e6, 'cd': 5e-1}
                }
    heddleson_re = []
    heddleson_cd = []
    for test_name, test_values in test_dict.items():
        heddleson_cd += [test_values['cd']]
        heddleson_re += [test_values['reynolds']]

    heddleson_re = np.array(heddleson_re)
    heddleson_cd = np.array(heddleson_cd)
    return heddleson_re, heddleson_cd

def get_hoerner_datapoints():
    # digitized from Figure 12, page 52
    # Hoerner, 1965, 'Fluid-dynamic drag: Practical information on aerodynamic drag and hydrodynamic resistance'
    # https://home.hvl.no/ansatte/gste/ftp/MarinLab_files/Litteratur/Hoerner_1965_Fluid-dynamic_drag.pdf
    hoerner_re = [0.006173, 0.009123, 0.013397, 0.018716, 0.027289, 0.041078, 0.057483, 0.09077, 0.12406, 0.20339,
                  0.28933, 0.45739, 0.6891, 0.9569, 1.4471, 2.0938, 3.1738, 5.3203, 8.207, 11.094, 20.096, 29.732,
                  47.295, 74.98, 121.43, 199.93, 306.94, 535.38, 763.8, 1384.3, 2026.3, 3594, 5294.5, 8674, 14509,
                  22097, 39924, 58907, 96230, 156110, 244330, 331450, 397000, 430410, 661700, 1044400, 1716100, 2693300,
                  4267200, 6922000, 11241000, 18141000, 29143000, 46897000, 76260000, 121180000]
    hoerner_cd = [576.41, 425.31, 316.08, 222.07, 164.04, 122.79, 87.77, 68.16, 48.542, 38.026, 28.815, 22.13, 17.403,
                  12.905, 9.811, 7.125, 5.5688, 4.4299, 3.5806, 2.7313, 2.3349, 1.9856, 1.702, 1.5023, 1.3506, 1.2811,
                  1.2148, 1.1626, 1.1104, 1.0668, 1.0238, 1.0099, 1.0002, 1.1171, 1.179, 1.1889, 1.1885, 1.1869, 1.1724,
                  1.1547, 1.0489, 0.7736, 0.49213, 0.29929, 0.28629, 0.34109, 0.38136, 0.40619, 0.4268, 0.41404,
                  0.40852, 0.38419, 0.35967, 0.33987, 0.3095, 0.28001]
    return hoerner_re, hoerner_cd

def get_yuce_datapoints():
    # Yuce and Kareem,
    # A Numerical Analysis of Fluid Flow Around Circular and Square Cylinders,
    #     http://dx.doi.org/10.5942/jawwa.2016.108.0141
    #     2016 Journal of American Water Works Association
    yuce_re = [2., 4., 15., 38, 160, 190, 250, 290, 1e3, 1e4, 1e5, 2.5e5, 1e6, 1.5e6, 2e6, 4e6]
    yuce_cd = [11.72, 7.16, 3.08, 1.91, 1.36, 1.34, 1.33, 1.38, 1.31, 0.89, 0.64, 0.515, 0.51, 0.51, 0.552, 0.47]
    return yuce_re, yuce_cd

def get_achenbach_datapoints():

    # data 3.9376e4 < Re < 4.815676e6, from fig 9, pg. 635 of
    # Achenbach, E. (1968). Distribution of local pressure and skin friction around a circular cylinder in cross-flow up to Re = 5e6
    # Journal of Fluid Mechanics, 34(4), 625-639. doi:10.1017/S0022112068002120
    # digitized by Mikko Folkersma (TU Delft, 2017)

    achenbach_re = np.array([39376.9444845985,
                            59180.3220925593,
                            98042.1284988708,
                            130950.661845122,
                            177279.465034671,
                            189171.336716672,
                            208523.702821741,
                            219382.685340574,
                            250474.492764236,
                            251176.102207476,
                            272662.507546918,
                            293391.667767627,
                            297121.039888131,
                            329418.920869394,
                            356054.506766152,
                            430668.007087765,
                            442913.779559282,
                            474051.811029239,
                            563061.279073674,
                            639260.260850478,
                            744111.718331051,
                            856111.913874845,
                            975544.366401701,
                            1284502.83554852,
                            1521822.48808959,
                            1545281.10611444,
                            1808117.79265064,
                            2165145.64882329,
                            2946433.47747528,
                            3600828.01647156,
                            4815676.41745568])

    achenbach_cd = np.array([1.2699208416,
                            1.2699208416,
                            1.1926111128,
                            1.0424288576,
                            0.9132048434,
                            0.7512979227,
                            0.8889507915,
                            0.7822542222,
                            0.6378155003,
                            0.7597718936,
                            0.6822150289,
                            0.6003319778,
                            0.4136881663,
                            0.540261156,
                            0.6003319778,
                            0.5318440363,
                            0.6030313516,
                            0.7280702551,
                            0.6566890304,
                            0.6003319778,
                            0.6776394186,
                            0.6335376772,
                            0.6307017467,
                            0.5976446873,
                            0.6992581883,
                            0.581771682,
                            0.6112025174,
                            0.7597718936,
                            0.7805014403,
                            0.7614781229,
                            0.7346324652])

    return achenbach_re, achenbach_cd

def make_poly_points(smoothing):

    number_points = 200.
    log_reynolds_list = 2. +  np.arange(number_points + 1.) * 5. / number_points
    cd_list = []
    reynolds_list = []

    for log_reynolds in log_reynolds_list:
        reynolds = 10.**log_reynolds
        reynolds_list = cas.vertcat(reynolds_list, reynolds)

        cd_list = cas.vertcat(cd_list, get_roshko_unitstep(reynolds, smoothing))

    cd_list = np.array(cd_list)
    reynolds_list = np.array(reynolds_list)

    return reynolds_list, cd_list

def get_interpolation(reynolds, smoothing):

    # [reynolds_array, cd_array] = make_poly_points(smoothing)
    hoerner_re, hoerner_cd = get_hoerner_datapoints()
    reynolds_array = np.array(hoerner_re)
    cd_array = np.array(hoerner_cd)

    min_re = 1e3
    max_re = 1e7
    log_re_list = []
    cd_list = []
    for idx in range(len(hoerner_re)):
        local_re = hoerner_re[idx]
        if local_re < max_re and local_re > min_re:
            log_re_list += [np.log10(local_re)]
            cd_list += [cd_array[idx]]

    max_dim = 5

    poly = np.polyfit(log_re_list, cd_list, max_dim)

    estimate = cas.polyval(poly, np.log10(reynolds))

    return estimate

if __name__ == "__main__":
    plot_cd_vs_reynolds(1)
    test()

    import awebox.tools.print_operations as print_op
    re_sym = cas.SX.sym('re_sym')
    cdp_list = get_cd_piecewise_info_list(re_sym, cd_outside_bounds=1e3)
    cdp_dict = {}
    for idx in range(len(cdp_list)):
        cdp_dict[idx] = cdp_list[idx]
    print_op.print_dict_as_table(cdp_dict, level='info', transpose=True)