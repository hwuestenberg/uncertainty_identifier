# ###################################################################
# script data_converter
#
# Description
# Convert raw data into general file format.
#
# ###################################################################
# Author: hw
# created: 07. Apr. 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
import sys

from uncert_ident.data_handling.dataconverter import dataConverter


#####################################################################
### Import config
#####################################################################
sys.path.append("..")


#####################################################################
### Choose all data sets or specific sets
#####################################################################
# Process all data sets
flag_ALL = False

# Separation
flag_CDC_Laval = False
flag_CBFS_Bentaleb = True
flag_PH_Breuer = True
flag_PH_Xiao = True
flag_Cube_Rossi = False  # Not implemented

# Complex flows
flag_JiCF_Milani = False  # Not implemented

# Pressure gradients
flag_TBL_APG_Bobke = True
flag_NACA4412_Vinuesa = True
flag_NACA0012_Tanarro = True

# Streamline curvature
flag_BP_Noorani = False



#####################################################################
### Case setup
#####################################################################
# CDC-Laval
if flag_ALL or flag_CDC_Laval:
    laval = dataConverter('CDC-Laval', 'dat', 1)
    laval.set_parameters(dimension=2,
                         coord_sys='cartesian',
                         geometry='converging_diverging_channel',
                         geometry_scale=1,
                         char_length=1,  # Half channel height
                         kin_visc=1 / 12600,
                         density=1000,  # Assume water
                         ghost_nx=0,  # No ghost cells
                         block_order=[],  # No blocks
                         block_ranges=[],
                         subcase_names=[],  # No subcases
                         subcase_nxs=2304,
                         subcase_nys=385,
                         csv_names=['conv-div-mean',
                                    'conv-div-budgets'],
                         col_names=[
                             ['x', 'y',
                              'um', 'vm', 'wm',
                              'grad_du_dx', 'grad_dv_dx', 'grad_dw_dx',
                              'grad_du_dy', 'grad_dv_dy', 'grad_dw_dy',
                              'grad_du_dz', 'grad_dv_dz', 'grad_dw_dz',
                              'uu', 'uv', 'uw', 'vv', 'vw', 'ww'],
                             ['x', 'y',
                              'diss_uu', 'diss_uv', 'diss_uw', 'diss_vv', 'diss_vw', 'diss_ww',
                              'prs_str_uu', 'prs_str_uv', 'prs_str_uw', 'prs_str_vv', 'prs_str_vw', 'prs_str_ww',
                              'prs_dif_uu', 'prs_dif_uv', 'prs_dif_uw', 'prs_dif_vv', 'prs_dif_vw', 'prs_dif_ww',
                              'prod_uu', 'prod_uv', 'prod_uw', 'prod_vv', 'prod_vw', 'prod_ww',
                              'conv_uu', 'conv_uv', 'conv_uw', 'conv_vv', 'conv_vw', 'conv_ww',
                              'vsc_dif_uu', 'vsc_dif_uv', 'vsc_dif_uw', 'vsc_dif_vv', 'vsc_dif_vw', 'vsc_dif_ww',
                              'trb_tsp_uu', 'trb_tsp_uv', 'trb_tsp_uw', 'trb_tsp_vv', 'trb_tsp_vw', 'trb_tsp_ww']],
                         csv_skip_headers=[20+6, 44+6],  # Often 6 lines for tecplot style files
                         csv_delimiter='',
                         manipulation_args=[
                             [['diss_uu', 'diss_uv', 'diss_uw', 'diss_vv', 'diss_vw', 'diss_ww'], 12600 / 2, ],
                             [['vsc_dif_uu', 'vsc_dif_uv', 'vsc_dif_uw', 'vsc_dif_vv', 'vsc_dif_vw', 'vsc_dif_ww'],
                              12600]]
                         )
    laval.run()


# CBFS-Bentaleb
if flag_ALL or flag_CBFS_Bentaleb:
    bentaleb = dataConverter('CBFS-Bentaleb', 'dat', 1)
    bentaleb.set_parameters(dimension=2,
                            coord_sys='cartesian',
                            geometry='curved_backwards_facing_step',
                            geometry_scale=1,
                            char_length=1,  # Step height
                            kin_visc=1/12600,  # Assume value from Laval
                            density=1000,  # Assume water
                            ghost_nx=0,  # No ghost cells
                            block_order=[],  # No blocks
                            block_ranges=[],
                            subcase_names=[],  # No subcases
                            subcase_nxs=768,
                            subcase_nys=160,
                            csv_names=['curvedbackstep_vel_stress',
                                       'curvedbackstep_vel_derivs',
                                       'curvedbackstep_budgets_all'],
                            col_names=[
                                ['x', 'y',
                                 'pm', 'um', 'vm', 'wm',
                                 'uu', 'vv', 'ww', 'uv', 'uw', 'vw',
                                 'k'],
                                ['x', 'y',
                                 'grad_du_dx', 'grad_du_dy',
                                 'grad_dv_dx', 'grad_dv_dy',
                                 'grad_dw_dx', 'grad_dw_dy'],
                                ['x', 'y',
                                 'prod_k', 'trb_tsp_k', 'prs_str_k', 'prs_dif_k', 'vsc_dif_k', 'diss_k', 'conv_k',
                                 'prod_uu', 'trb_tsp_uu', 'prs_str_uu', 'prs_dif_uu', 'vsc_dif_uu', 'diss_uu', 'conv_uu',
                                 'prod_uv', 'trb_tsp_uv', 'prs_str_uv', 'prs_dif_uv', 'vsc_dif_uv', 'diss_uv', 'conv_uv',
                                 'prod_vv', 'trb_tsp_vv', 'prs_str_vv', 'prs_dif_vv', 'vsc_dif_vv', 'diss_vv', 'conv_vv',
                                 'prod_ww', 'trb_tsp_ww', 'prs_str_ww', 'prs_dif_ww', 'vsc_dif_ww', 'diss_ww', 'conv_ww']
                            ],
                            csv_skip_headers=[13+6, 8+6, 37+6],  # Often 6 lines for tecplot style files
                            csv_delimiter='',
                            )
    bentaleb.run()


# PH-Breuer
if flag_ALL or flag_PH_Breuer:
    breuer = dataConverter('PH-Breuer', 'csv', 5)
    breuer.set_parameters(dimension=2,
                          coord_sys='cartesian',
                          geometry='periodic_hills',
                          geometry_scale=1,
                          char_length=1,  # Hill height
                          kin_visc=1/12600,
                          density=1000,  # kg/m^3
                          ghost_nx=3,
                          block_order=[0, 1, 2, 3, 4, 7, 6, 5],
                          block_ranges=[range(0, 8711),
                                        range(8711, 16017),
                                        range(16017, 24728),
                                        range(24728, 33439),
                                        range(33439, 41588),
                                        range(41588, 50299),
                                        range(50299, 57605),
                                        range(57605, 65754)],
                          subcase_names=['PH-Breuer-700',
                                         'PH-Breuer-1400',
                                         'PH-Breuer-2800',
                                         'PH-Breuer-5600',
                                         'PH-Breuer-10595'
                                         ],
                          subcase_nxs=281,
                          subcase_nys=220,
                          subcase_path_to_raw=['Re_700/',
                                               'Re_1400/',
                                               'Re_2800/',
                                               'Re_5600/',
                                               'Re_10595/'
                                               ],
                          csv_names=['Hill_Breuer'],  # Equal name for all subcases
                          col_names=[['x', 'y',
                                      'um', 'vm', 'wm', 'pm',
                                      'uu', 'vv', 'ww', 'uv']],
                          csv_skip_headers=1,
                          csv_delimiter=',',
                          csv_newline_char=',\n',
                          )
    breuer.run()


# PH-Xiao
if flag_ALL or flag_PH_Xiao:
    xiao = dataConverter('PH-Xiao', 'dat', 5)
    xiao.set_parameters(dimension=2,
                        coord_sys='cartesian',
                        geometry='periodic_hills',
                        geometry_scale=[0.5, 0.8, 1.0, 1.2, 1.5],
                        char_length=1,
                        kin_visc=1/12600,  # Assume water, value from CDC-Laval
                        density=1000,  # kg/m^3
                        ghost_nx=0,  # No ghost cells
                        block_order=[],  # No blocks
                        block_ranges=[],
                        subcase_names=['PH-Xiao_05',
                                       'PH-Xiao_08',
                                       'PH-Xiao_10',
                                       'PH-Xiao_12',
                                       'PH-Xiao_15'
                                       ],
                        subcase_nxs=[736, 704, 768, 832, 934],
                        subcase_nys=385,
                        subcase_path_to_raw=['case_0p5/dns-data',
                                             'case_0p8/dns-data',
                                             'case_1p0_refined_XYZ/dns-data',  # Use only the refined case
                                             'case_1p2/dns-data/',
                                             'case_1p5/dns-data'
                                             ],
                        csv_names=['mean_files',
                                   'rms_files1',
                                   'rms_files2'],
                        col_names=[
                            ['x', 'y', 'um', 'vm', 'wm', 'pm'],
                            ['x', 'y', 'uu', 'vv', 'ww', 'pp'],
                            ['x', 'y', 'uv', 'uw', 'vw']
                        ],
                        csv_skip_headers=[1, 1, 1],
                        csv_delimiter='',
                        csv_newline_char=' \n',
                        solid_grid_points_to_zero=True
                        )
    xiao.run()


# Cube-Rossi
#  TODO Read in Rossi's wmc data, probably requires interpolation
if flag_ALL or flag_Cube_Rossi:
    pass


# JiCF-Milani
#  TODO Read in milani's JICF data, requires identification of block-structures, Huge data set >80 million points
#  Out of memory error (code 137), may use Pandas with chunk-wise csv reading
if flag_ALL or flag_JiCF_Milani:
    pass


# TBL-APG-Bobke
if flag_ALL or flag_TBL_APG_Bobke:
    bobke = dataConverter('TBL-APG-Bobke', 'mat', 5)
    bobke.set_parameters(dimension=2,
                         coord_sys='cartesian',
                         geometry='flat_plate',
                         geometry_scale=1,
                         char_length='delta99',
                         kin_visc=1/12600,
                         density=1000,  # kg/m^3
                         ghost_nx=0,
                         block_order=[],
                         block_ranges=[],
                         subcase_names=['TBL-APG-Bobke-b1',
                                        'TBL-APG-Bobke-b2',
                                        'TBL-APG-Bobke-m13',
                                        'TBL-APG-Bobke-m16',
                                        'TBL-APG-Bobke-m18',
                                        ],
                         subcase_nxs=1570,
                         # subcase_nys=[301, 361, 301, 361, 401],  # Full grid
                         subcase_nys=[201, 241, 201, 241, 241],  # Remove top section (and b2)
                         mat_name='APG',
                         )
    bobke.run()


# NACA4412-Vinuesa
if flag_ALL or flag_NACA4412_Vinuesa:
    naca4412 = dataConverter('NACA4412-Vinuesa', 'mat', 8)
    naca4412.set_parameters(dimension=2,
                            coord_sys='cartesian',
                            geometry='naca4412',
                            geometry_scale=1,
                            char_length='delta99',
                            wall_offset='ya',
                            kin_visc=1/12600,
                            density=1000,  # kg/m^3
                            ghost_nx=0,
                            block_order=[],
                            block_ranges=[],
                            subcase_names=['NACA4412-Vinuesa-top-1',
                                           'NACA4412-Vinuesa-top-2',
                                           'NACA4412-Vinuesa-top-4',
                                           'NACA4412-Vinuesa-top-10',
                                           'NACA4412-Vinuesa-bottom-1',
                                           'NACA4412-Vinuesa-bottom-2',
                                           'NACA4412-Vinuesa-bottom-4',
                                           'NACA4412-Vinuesa-bottom-10',
                                           ],
                            subcase_nxs=[50],
                            subcase_nys=[52, 52, 52, 61, 52, 52, 52, 61],
                            mat_name='naca4412',
                            )
    naca4412.run()


# NACA0012-Tanarro
if flag_ALL or flag_NACA0012_Tanarro:
    naca0012 = dataConverter('NACA0012-Tanarro', 'mat', 1)
    naca0012.set_parameters(dimension=2,
                            coord_sys='cartesian',
                            geometry='naca0012',
                            geometry_scale=1,
                            char_length='delta99',
                            wall_offset='ya',
                            kin_visc=1/12600,
                            density=1000,  # kg/m^3
                            ghost_nx=0,
                            block_order=[],
                            block_ranges=[],
                            subcase_names='NACA0012-Tanarro-top-4',
                            subcase_nxs=50,
                            subcase_nys=52,
                            mat_name='naca0012',
                            )
    naca0012.run()


# BP-Noorani
if flag_ALL or flag_BP_Noorani:
    noorani = dataConverter('BP-Noorani', 'dat', 3)
    noorani.set_parameters(dimension=2,
                           coord_sys='cylinder',
                           geometry='bent_pipe',
                           geometry_scale=1,
                           char_length=1,  # Pipe radius
                           kin_visc=1/12600,  # Assume water, value from CDC-Laval
                           density=1000,  # kg/m^3
                           ghost_nx=0,  # No ghost cells
                           block_order=[],  # No blocks
                           block_ranges=[],
                           subcase_names=['BP-Noorani-11700-001',
                                          'BP-Noorani-11700-01',
                                          # 'BP-Noorani-11700-03', # Data does not converge in cubic eddy viscosity
                                          'BP-Noorani-5300-001',
                                          ],
                           subcase_nxs=360,  # Number of cells in angular direction
                           subcase_nys=315,  # Number of cells in radial direction
                           subcase_path_to_raw=['R11700k001',
                                                'R11700k01',
                                                # 'R11700k03',
                                                'R5300k001'
                                                ],
                           csv_names=['1stAND2ndOrderMoments',
                                      'BudgetR11',
                                      'BudgetR22',
                                      'BudgetR33',
                                      'MeanDerivat'
                                      ],
                           col_names=[['x', 'y',
                                       'urm', 'utm', 'usm', 'pm',
                                       'ur_rms', 'ut_rms', 'us_rms', 'p_rms'],
                                      ['x', 'y',
                                       'conv_urur', 'prod_urur', 'diss_urur',
                                       'prs_str_urur', 'prs_dif_urur', 'vsc_dif_urur',
                                       'trb_tsp_urur', 'balance_urur'],
                                      ['x', 'y',
                                       'conv_utut', 'prod_utut', 'diss_utut',
                                       'prs_str_utut', 'prs_dif_utut', 'vsc_dif_utut',
                                       'trb_tsp_utut', 'balance_utut'],
                                      ['x', 'y',
                                       'conv_usus', 'prod_usus', 'diss_usus',
                                       'prs_str_usus', 'prs_dif_usus', 'vsc_dif_usus',
                                       'trb_tsp_usus', 'balance_usus'],
                                      ['x', 'y',
                                       'grad_dur_dr', 'grad_dur_dt', 'grad_dur_ds',
                                       'grad_dut_dr', 'grad_dut_dt', 'grad_dut_ds',
                                       'grad_dus_dr', 'grad_dus_dt', 'grad_dus_ds']
                                      ],
                           csv_skip_headers=23,
                           csv_delimiter='',
                           )
    noorani.run()
# TODO Incomplete tauij for all cases, wait for response by P. Schlatter
# TODO BP-Noorani-11700-03 does not converge in cubic_tauij; data seems to be corrupted (derivs dont fit to mean data)





print("\nEOF convert_to_mat.py")
