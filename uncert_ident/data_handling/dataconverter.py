# ###################################################################
# class dataConverter
#
# Description
# Class for conversion of distinct high-fidelity data into common 
# data format.
#
# ###################################################################
# Author: hw
# created: 17. May 2020
# ###################################################################
#####################################################################
### Import
#####################################################################
from os.path import basename, splitext
import sys
import numpy as np
from timeit import default_timer as timer
from copy import deepcopy

from uncert_ident.data_handling import data_import as di
from uncert_ident.visualisation import plotter as plot
from uncert_ident.utilities import DATASET_PARAMETERS, get_shape, safe_divide, TAU_KEYS, append_list_till_length
from uncert_ident.methods.turbulence_properties import turbulence_properties
from uncert_ident.methods.geometry import geometry_periodic_hills_lower


#####################################################################
### Class
#####################################################################
class dataConverter:
    def __init__(self, case_name, filetype, num_of_subcases=1):
        """
        Define case name according to table "DATA/data_overview". And
        type of data files.
        """

        assert (type(case_name)) is str, "case_name is not a string: %r" % case_name
        self.case_name = case_name
        self.path_to_data = di.path_to_raw_data + case_name + '/'

        # Define file extension for input files
        if filetype == 'csv':
            self.file_ext = '.csv'
        elif filetype == 'dat':
            self.file_ext = '.dat'
        elif filetype == 'mat':
            self.file_ext = '.mat'

        # Define amount of subcases
        self.num_of_subcases = num_of_subcases

        # Declare common dictionary
        self.maindict = dict()


    def set_parameters(self,
                       dimension, coord_sys, geometry, geometry_scale, char_length, kin_visc, density,
                       ghost_nx, block_order, block_ranges,
                       subcase_names, subcase_nxs, subcase_nys, parameter_adds=None,
                       manipulation_args=None, solid_grid_points_to_zero=False,
                       wall_offset=0,
                       csv_names=None, col_names=None, mat_name=None, subcase_path_to_raw=None,
                       csv_skip_headers=1, csv_delimiter=',', csv_newline_char=None):
        """
        Set all required parameters
        :param dimension: See setter-functions.
        :param coord_sys: ""
        :param geometry: ""
        :param geometry_scale: ""
        :param char_length: ""
        :param kin_visc: ""
        :param density: ""
        :param wall_offset: ""
        :param ghost_nx: ""
        :param block_order: ""
        :param block_ranges: ""
        :param subcase_names: ""
        :param subcase_nxs: ""
        :param subcase_nys: ""
        :param parameter_adds: ""
        :param manipulation_args: ""
        :param solid_grid_points_to_zero: ""
        :param csv_names:  ""
        :param col_names: ""
        :param mat_name: ""
        :param subcase_path_to_raw: ""
        :param csv_skip_headers: ""
        :param csv_delimiter: ""
        :param csv_newline_char: ""
        :return: 1:success.
        """

        # General parameters
        self.set_dimension(dimension)
        self.set_coord_sys(coord_sys)
        self.set_geometry(geometry)
        self.set_geometry_scale(geometry_scale)
        self.set_char_length(char_length)
        self.set_kinematic_viscosity(kin_visc)
        self.set_density(density)
        self.set_wall_offset(wall_offset)

        # Grid properties
        self.set_num_ghost_nx(ghost_nx)
        self.set_block_ranges(block_ranges)
        self.set_block_order(block_order)

        # Subcases
        self.set_subcase_names(subcase_names)
        self.set_subcase_nxs(subcase_nxs)
        self.set_subcase_nys(subcase_nys)
        self.set_parameter_adds(parameter_adds)
        self.set_manipulation_arguments(manipulation_args)
        self.set_solid_grid_points_to_zero(solid_grid_points_to_zero)

        # Filenames/paths
        self.set_subcase_path_to_raw(subcase_path_to_raw)

        if csv_names:
            self.set_csv_names(csv_names)
            self.set_col_names(col_names)

            # csv properties
            self.set_csv_skip_headers(csv_skip_headers)
            self.set_csv_delimiter(csv_delimiter)
            self.set_csv_newline_char(csv_newline_char)

        elif mat_name:
            self.set_mat_name(mat_name)

        else:
            print('No filenames (and column names) were defined. Define with setter-functions.')


        return 1


    def set_dimension(self, dim):
        """
        Set the dimension parameter.
        :return: 1:success.
        """

        self.dimension = dim

        return 1

    def set_coord_sys(self, csys):
        """
        Set the coordinate system.
        :return: 1:success.
        """

        self.coord_sys = csys

        return 1

    def set_geometry(self, geo):
        """
        Set the geometry.
        :return: 1:success.
        """

        self.geometry = geo

        return 1

    def set_geometry_scale(self, geo_scale):
        """
        Set the scaling factor for the geometry.
        :return: 1:success.
        """

        if isinstance(geo_scale, (int, float)):
            geo_scale = list([geo_scale])
            self.geometry_scale = append_list_till_length(geo_scale, self.num_of_subcases)

        elif isinstance(geo_scale, list) and len(geo_scale) == self.num_of_subcases:
            self.geometry_scale = geo_scale

        elif isinstance(geo_scale, list) and len(geo_scale) < self.num_of_subcases:
            self.geometry_scale = append_list_till_length(geo_scale, self.num_of_subcases)

        else:
            assert False, 'Invalid geometry scale for subcases: %r' % geo_scale

        return 1

    def set_char_length(self, length):
        """
        Set the characteristic length as floating
        point number or as string of a varying
        quantity in data. The string must conform with
        a string in mat file or in col_names.
        :return: 1:success.
        """

        self.char_length = length

        return 1

    def set_kinematic_viscosity(self, nu):
        """
        Set the kinematic viscosity of the
        simulated fluid.
        :return: 1:success.
        """

        self.nu = nu

        return 1

    def set_density(self, rho):
        """
        Set the density of the simulated fluid.
        :return: 1:success.
        """

        self.rho = rho

        return 1

    def set_wall_offset(self, wall_off):
        """
        Set the density of the simulated fluid.
        :return: 1:success.
        """

        self.wall_offset = wall_off

        return 1

    def set_num_ghost_nx(self, ghost_nx):
        """
        Set the number of ghost cells in x-direction.
        Symmetric ghost cells are assumed.
        :return: 1:success.
        """

        self.num_ghost_nx = ghost_nx

        return 1

    def set_block_order(self, blck_ord):
        """
        Set the order of block grids assuming 0
        is the bottom-most block.
        :return: 1:success.
        """

        self.block_order = blck_ord

        return 1

    def set_block_ranges(self, blck_rngs):
        """
        Set the range of each block in terms of indices
        in flat arrays.
        :return: 1:success.
        """

        self.block_ranges = blck_rngs

        return 1

    def set_subcase_names(self, names):
        """
        Set individual names for each subcase according to
        ta table data_overview.
        :return: 1:success.
        """

        if isinstance(names, list) and len(names) > 0:
            self.subcase_names = names
        elif isinstance(names, list) and len(names) == 0:
            self.subcase_names = list([self.case_name])
        elif isinstance(names, str):
            self.subcase_names = list([names])

        else:
            assert False, 'Invalid subcase names: %r' % names

        assert len(self.subcase_names) == self.num_of_subcases, 'Invalid amount of subcase_names.'

        return 1

    def set_parameter_adds(self, adds):
        """
        NOT IN USE ANYMORE
        Set individual additions for each subcase name
        according to data_overview.ods table.
        Required for saving processed mat files.
        :return: 1:success.
        """

        if isinstance(adds, list) and len(adds) > 0:
            self.parameter_adds = adds
        elif (isinstance(adds, list) and len(adds) == 0) or adds == None:
            self.parameter_adds = list([''])
        elif isinstance(adds, str):
            self.parameter_adds = list([adds])

        else:
            assert False, 'Invalid parameter adds for subcases: %r' % adds

        return 1

    def set_subcase_nxs(self, nxs):
        """
        Set nx for each subcase.
        :return: 1:success.
        """

        if isinstance(nxs, (int, float)):
            nxs = list([nxs])
            self.subcase_nxs = append_list_till_length(nxs, self.num_of_subcases)

        elif isinstance(nxs, list) and len(nxs) == self.num_of_subcases:
            self.subcase_nxs = nxs

        elif isinstance(nxs, list) and len(nxs) < self.num_of_subcases:
            self.subcase_nxs = append_list_till_length(nxs, self.num_of_subcases)

        else:
            assert False, 'Invalid nxs for subcases: %r' % nxs

        return 1

    def set_subcase_nys(self, nys):
        """
        Set ny for each subcase.
        :param nys: Number of points in y-direction.
        :return: 1:success.
        """

        if isinstance(nys, (int, float)):
            nys = list([nys])
            self.subcase_nys = append_list_till_length(nys, self.num_of_subcases)

        elif isinstance(nys, list) and len(nys) == self.num_of_subcases:
            self.subcase_nys = nys

        elif isinstance(nys, list) and len(nys) < self.num_of_subcases:
            self.subcase_nys = append_list_till_length(nys, self.num_of_subcases)

        else:
            assert False, 'Invalid nys for subcases: %r' % nys

        return 1

    def set_manipulation_arguments(self, man_args):
        """
        Set manipulation of keys and factors in list of
        key-factor pairs according to the manipulate_subdicts
        function. Can be None.
        :param man_args: List of key-factor pairs.
        :return: 1:success.
        """

        self.manipulation_args = man_args

        return 1

    def set_solid_grid_points_to_zero(self, points_to_zero):
        """
        Set boolean option whether all fields are set to zero when
        detected to lie inside a boundary.
        :param points_to_zero: Boolean.
        :return: 1:success.
        """

        self.solid_grid_points_to_zero = points_to_zero

        return 1

    def set_csv_names(self, names):
        """
        Define csv_names that contain the data WIHTOUT
        file extension.
        :param names: List of filenames.
        :return: 1:success.
        """

        self.csv_names = names

        return 1

    def set_mat_name(self, name):
        """
        Define name of the mat file that contains the data
        WIHTOUT file extension.
        :param name: Filename.
        :return: 1:success.
        """

        self.mat_name = name

        return 1

    def set_col_names(self, cols):
        """
        Define variable names for each column in the csv_files.
        :param col_names: List of a list for each file with names.
        :return: 1:success.
        """

        self.col_names = cols

        return 1

    def set_subcase_path_to_raw(self, path_to_raw):
        """
        Set additional path for subcase files
        Optional for reading subcase files/paths.
        :return: 1:success.
        """

        if path_to_raw == None or path_to_raw == []:
            path_to_raw = list([''])
            self.subcase_path_to_raw = append_list_till_length(path_to_raw, self.num_of_subcases)

        elif isinstance(path_to_raw, str):
            # Add / if not defined
            if not path_to_raw[-1] == '/':
                self.subcase_path_to_raw = list([path_to_raw + '/'])
            else:
                self.subcase_path_to_raw = list([path_to_raw])

        elif isinstance(path_to_raw, list):
            self.subcase_path_to_raw = list()
            for path in path_to_raw:
                # Add / if not defined
                if not path[-1] == '/':
                    path = path + '/'
                else:
                    pass
                self.subcase_path_to_raw.append(path)
            self.subcase_path_to_raw = append_list_till_length(self.subcase_path_to_raw, self.num_of_subcases)

        else:
            assert False, 'Ivalid path addition: %r' % path_to_raw

        return 1

    def set_csv_skip_headers(self, skip_headers):
        """
        Set amount of lines to skip from the top
        when reading the csv/dat file.
        :return: 1:success.
        """

        if isinstance(skip_headers, (int, float)):
            skip_headers = list([skip_headers])
            self.csv_skip_headers = append_list_till_length(skip_headers, len(self.csv_names))

        elif isinstance(skip_headers, list) and len(skip_headers) == len(self.csv_names):
            self.csv_skip_headers = skip_headers

        elif isinstance(skip_headers, list) and len(skip_headers) < len(self.csv_names):
            self.csv_skip_headers = append_list_till_length(skip_headers, len(self.csv_names))

        else:
            assert False, 'Invalid csv_skip_headers. Must be list, int or float, but given: %r' % skip_headers

        return 1

    def set_csv_delimiter(self, delimiter):
        """
        Set delimiter for the csv/dat files.
        :return: 1:success.
        """

        self.csv_delimiter = delimiter

        return 1

    def set_csv_newline_char(self, newline_char):
        """
        Set delimiter for the csv/dat files.
        :return: 1:success.
        """

        self.csv_newline_char = newline_char

        return 1


    def check_file_available(self):
        """
        Check whether the requested file can be
        found under the given path.
        :return: 1:file found, 0: file not found.
        """

        if self.check_individual_parameter_defined:
            for subcase_path_to_raw in self.subcase_path_to_raw:

                if self.file_ext == '.csv' or self.file_ext == '.dat':
                    # Add check_extension function
                    for csv_name in self.csv_names:
                        if di.exist_file(self.path_to_data + subcase_path_to_raw + csv_name + self.file_ext):  # Does not recognise if subfolder is different, but filename equivalent
                            pass
                        else:
                            print('Cannot find file: %r' % self.path_to_data + subcase_path_to_raw + csv_name + self.file_ext)
                            return 0

                # Assume only one mat file is required
                elif self.file_ext == '.mat':
                    di.check_mat_extension(self.mat_name, add_extension=False)
                    if di.exist_file(self.mat_name + self.file_ext):
                        pass
                    else:
                        print('Cannot find file: %r' % self.path_to_data + self.mat_name + self.file_ext)
                        return 0

        return 1

    def check_all_parameter_defined(self):
        """
        Check whether all parameters necessary for conversion have been
        defined. If so, the main method can be called to convert the
        data.
        :param self: Object of class dataconverter
        :return: 1:ready for conversion. 0:Not all parameters defined.
        """

        for param in DATASET_PARAMETERS:
            undefined_params = list()
            if not hasattr(self, param):
                undefined_params.append(param)

            if len(undefined_params) > 0:
                print('The following parameters need to be defined before running the converter:')
                print(undefined_params)
                print('Use set_parameters to define all parameters or indivdiual setter functions.')
                return 0

            else:
                # print('All parameters have been defined.')
                return 1

    def check_individual_parameter_defined(self, param):
        """
        Check whether all parameters necessary for conversion have been
        defined. If so, the main method can be called to convert the
        data.
        :param self: Object of class dataconverter
        :param param: Parameter in class dataConverter.
        :return: 1:Parameter defined. 0:Undefined parameter.
        """

        if not hasattr(self, param):
            print('Undefined parameter: %r' % param)
            print('Use set_parameters to define all parameters or indivdiual setter functions.')
            return 0

        else:
            return 1

    def get_subdicts(self):
        """
        Generate an iterable of links to subcase_dictionaries.
        :return: List of subdicts.
        """

        subdicts = list()
        for key in self.maindict:
            element = self.maindict[key]
            if isinstance(element, dict):
                subdicts.append(element)

        return subdicts


    def read_subdicts(self):
        # Check whether conversion is possible
        if not self.check_all_parameter_defined():
            sys.exit('Not all parameters have been defined. Use the set_parameters function.')

        if not self.check_file_available():
            sys.exit('Files could not be found. Check given csv_names or mat_name.')


        # Read csv data
        if self.file_ext == '.csv' or self.file_ext == '.dat':

            # Define each subcase and save into maindict
            for subcase_name, subcase_nx, subcase_ny, subcase_path_to_raw, geometry_scale in zip(self.subcase_names,
                                                                                                 self.subcase_nxs,
                                                                                                 self.subcase_nys,
                                                                                                 self.subcase_path_to_raw,
                                                                                                 self.geometry_scale):

                # Generate path to each file
                fnames = list()
                for csv_name in self.csv_names:
                    fnames.append(di.path_to_raw_data +
                                  self.case_name + '/' +
                                  'data/' +
                                  subcase_path_to_raw +
                                  csv_name + self.file_ext)

                # Read each file and write into subdict
                clock2 = timer()
                subdict = dict()
                for fname, col_name, csv_skip_header in zip(fnames, self.col_names, self.csv_skip_headers):
                    try:
                        subdict.update(di.load_csv(fname, col_name,
                                                   skip_header=csv_skip_header,
                                                   delimiter=self.csv_delimiter,
                                                   newline_char=self.csv_newline_char))
                    except OSError:
                        print('File %r could not be found and is skipped.' % basename(fname))


                read_time = np.round(timer() - clock2, 2)
                print(subcase_name + ', read data in \t' + str(read_time) + ' seconds')

                # Set subcase parameters
                subdict['dimension'] = self.dimension
                subdict['coord_sys'] = self.coord_sys
                subdict['geometry'] = self.geometry
                subdict['geometry_scale'] = geometry_scale
                subdict['char_length'] = self.char_length
                subdict['wall_offset'] = self.wall_offset
                subdict['nu'] = self.nu
                subdict['rho'] = self.rho
                subdict['nx'] = subcase_nx
                subdict['ny'] = subcase_ny
                subdict['num_of_points'] = subcase_nx*subcase_ny

                # Safe subdicts into maindict
                self.maindict[subcase_name] = subdict


        elif self.file_ext == '.mat':

            # Path to file
            fname = di.path_to_raw_data + \
                    self.case_name + '/' + \
                    'data/' + \
                    self.mat_name + self.file_ext

            clock2 = timer()
            # Load mat file, assume kth_data, because only raw kth data comes in mat
            self.maindict = di.load_mat(fname, kth_data=True)

            read_time = np.round(timer() - clock2, 2)
            print(self.case_name + ', read data in \t' + str(read_time) + ' seconds')

            rename_subdicts(self.maindict, self.subcase_names)

            # Loop subdicts
            for subcase_name, subcase_nx, subcase_ny, subcase_path_to_raw, geometry_scale, subdict in zip(self.subcase_names,
                                                                                                          self.subcase_nxs,
                                                                                                          self.subcase_nys,
                                                                                                          self.subcase_path_to_raw,
                                                                                                          self.geometry_scale,
                                                                                                          self.get_subdicts()):

                # Rename keys in each subdict
                rename_kth_keys(subdict)

                # Set subcase parameters (copy n waste)
                subdict['dimension'] = self.dimension
                subdict['coord_sys'] = self.coord_sys
                subdict['geometry'] = self.geometry
                subdict['geometry_scale'] = geometry_scale
                subdict['char_length'] = self.char_length
                subdict['wall_offset'] = self.wall_offset
                if 'nu' not in subdict:
                    subdict['nu'] = self.nu
                if 'rho' not in subdict:
                    subdict['rho'] = self.rho
                subdict['nx'] = subcase_nx
                subdict['ny'] = subcase_ny
                subdict['num_of_points'] = subcase_nx * subcase_ny

                # Reduce y dimensions
                for key in subdict:
                    ele = subdict[key]
                    if isinstance(ele, np.ndarray):
                        if len(ele.shape) == 2 and ele.shape[1] != subcase_ny:
                            subdict[key] = ele[:, :subcase_ny]



                # Safe subdicts into maindict
                self.maindict[subcase_name] = subdict

        return 1

    def manipulate_subdicts(self, keys, factors):
        """
        Manipulate qunatities in all subdicts equally, before
        processing the data set.
        :param keys: Keys in subdicts. Can be a list of keys
        which is equal in length to factors
        :param factors: Factor by which key should be altered.
        :return: 1:success.
        """

        # Adjust iterables
        if not isinstance(keys, list):
            keys = list([keys])

        if not isinstance(factors, list):
            factors = list([factors])

        if len(factors) == 1 and len(keys) > 1:
            for i in range(len(keys) - 1):
                factors = factors + [factors[0]]

        # Get subdicts
        for i, subdict in enumerate(self.get_subdicts()):
            # Manipulate keys with factors
            for key, factor in zip(keys, factors):
                subdict[key] = subdict[key] * factor

            self.maindict[self.subcase_names[i]] = subdict

        return 1

    def handle_manipulation(self):
        """
        Apply manipulations to each subdict.
        :return: 1:success.
        """

        for args in self.manipulation_args:
            self.manipulate_subdicts(*args)

        return 1

    def debug_data_arrangemement(self):
        """
        Plot the data arrangement in 2D with colour gradient
        from first to last point.
        :return: 1:success.
        """

        # Plot arrangement for each subdict
        for subdict in self.get_subdicts():
            plot.scattering(subdict['x'], subdict['y'], np.arange(subdict['nx'] * subdict['ny']), scale=10,
                            alpha=0.5)
        plot.show()

        return 1

    def adjust_and_turbulence(self, debug_dry_run=0):
        """
        Adjust orientation of flat arrays, if necessary,
        manipulations for distinct coordinate systems and
        compute turbulence quantities.
        :param debug_dry_run: Optional debugging. Does not
        compute turbulence quantities (time intensive).
        :return: 1:success.
        """

        # Process each subdict individually
        for i, subdict in enumerate(self.get_subdicts()):

            print(self.subcase_names[i] + ', computing turbulence quantities and rearranging arrays...')

            # Generate dictionary of block-dictionaries
            blockdicts = generate_blockdicts(subdict, self.block_ranges)

            # Adjust dictionaries and compute turbulence properties
            clock3 = timer()
            for blockdict_name in blockdicts:
                blockdict = blockdicts[blockdict_name]
                get_nx_ny(blockdict)
                get_polar_coordinates(blockdict, self.coord_sys)
                set_char_length(blockdict, subdict['char_length'])
                set_offset_from_wall(blockdict, subdict['wall_offset'])
                check_tauij_sign(blockdict)
                generate_2d_coord(blockdict)
                flatten_all_arrays(blockdict)
                check_orientation(blockdict, self.coord_sys)
                remove_central_point(blockdict, self.coord_sys)
                if debug_dry_run:
                    blockdict['dummy'] = 'dummy'  # For debugging purposes
                else:
                    turbulence_properties(blockdict, self.coord_sys)
                remove_ghost_cells(blockdict, np.min(subdict['y']), np.max(subdict['y']), self.num_ghost_nx)
                if self.solid_grid_points_to_zero:
                    set_solid_grid_points_to_zero(blockdict, subdict['geometry'])

            # Print timing
            adjust_turb_time = np.round(timer() - clock3, 2)
            print(self.subcase_names[i] + ', computed turbulence quantities after \t' + str(adjust_turb_time) + ' seconds')

            # Merge all blocks according to block order
            mergedict = merge_data(blockdicts, self.block_order)
            subdict.update(mergedict)

        return 1

    def debug_visualise_tke(self):
        """
        Plot the tke in a 2D domain.
        :return: 1:success.
        """

        # Plot arrangement for each subdict
        for subdict in self.get_subdicts():
            plot.scattering(subdict['x'], subdict['y'], subdict['k'], scale=10, alpha=0.5)
        plot.show()

        return 1

    def save_subdicts_to_mat(self):
        """
        Save each subcase's data into a mat file.
        :return: 1:success.
        """

        # Save individual subdicts into mats
        for subdict, subcase_name in zip(self.get_subdicts(), self.subcase_names):
            matname = di.path_to_processed_data + \
                      self.case_name + '/' + \
                      'processed/' + \
                      subcase_name.replace('_', '-')

            di.save_dict_to_mat(matname, subdict)

        return 1


    def run(self,
            debug_data_arrangement=0,
            debug_visualise_tke=0,
            debug_dry_run=0):
        """
        Read data set, convert into defined format, compute turbulence
        properties, gradients, [...] and save as mat.
        :param debug_data_arrangement: Optional debugging plot of the data arrangement.
        :param debug_visualise_tke: Optional debugging plot of the computed tke.
        :param debug_dry_run: Skip computation of turbulence properties
        for software maintenance.
        :return: 1:successful conversion.
        """

        # Total timing
        clock1 = timer()
        print('Processing ' + self.case_name + '...')

        # Read raw data into dictionaries
        self.read_subdicts()

        # Manipulate raw data before processing
        if self.manipulation_args:
            self.handle_manipulation()

        # Visualise data arrangement
        if debug_data_arrangement:
            self.debug_data_arrangemement()

        # Adjust the flat arrays and compute turbulence properties
        self.adjust_and_turbulence(debug_dry_run)

        # Save processed data into mat files
        self.save_subdicts_to_mat()

        # Visualise turbulent kinetic energy
        if debug_visualise_tke:
            self.debug_visualise_tke()

        # Total timing
        case_time = np.round(timer() - clock1, 2)
        print(self.case_name + ' converted after ' + str(case_time) + ' seconds\n')

        return 1




def generate_blockdicts(dictionary, block_ranges):
    """
    Generate sub-dictionaries inside the main dictionary for each
    block in the block-structured grid data.
    :param dictionary: Dictionary of raw data for given flow case.
    :param block_ranges: Range for each block in flat raw data.
    :return: Updated dictionary of block dictionaries.
    """

    # Return dictionary of blockdict
    blockdicts = dict()

    # Check whether blocks can be created, eventually return all data as single block
    if block_ranges is None or len(block_ranges) <= 1:
        blockdict_name = 'block' + '0'
        blockdicts[blockdict_name] = deepcopy(dictionary)  # Shallow copy might be enough here
        return blockdicts

    # Create blockdicts according to index ranges in block_ranges
    for i in range(0, len(block_ranges)):
        blockdict_name = 'block' + str(i)  # Generate blockdict names
        blockdict = dict()

        for key in dictionary.keys():
            if not isinstance(dictionary[key], np.ndarray):  # Copy only field data
                continue
            blockdict[key] = dictionary[key][block_ranges[i]]  # Copy field data to blockdict

        # Define sub-grid size
        blockdict['nx'] = dictionary['nx']
        blockdict['ny'] = int(blockdict['x'].shape[0] / blockdict['nx'])
        blockdicts[blockdict_name] = blockdict  # Add blockdict to a common dictionary

    return blockdicts


def check_orientation(dictionary, coordinate_system):
    """
    Check the orientation of flat data. Required orientation when
    reshaped in 2D is x-coordinate increases in row-wise direction.
    :param coordinate_system: Specification of coordinate system.
    :param dictionary: Dictionary of flow data for one block.
    :return: (Adjusted) block-dictionary.
    """

    if coordinate_system == 'cylinder':
        print('WARNING No implementation for assessment of cylinder coordinates yet.')
        return dictionary

    flag_switch_nx_ny = 0
    flag_transpose = 0

    nx = dictionary['nx']
    ny = dictionary['ny']
    x_2d = dictionary['x'].reshape(nx, ny)
    y_2d = dictionary['y'].reshape(nx, ny)

    # Take absolute if only negative y coordinates
    if np.all(y_2d < 0):
        y_2d = np.abs(y_2d)

    min_x = np.min(x_2d.flatten()[0:nx])
    max_x = np.max(x_2d.flatten()[0:nx])
    min_y = np.min(y_2d.flatten()[0:ny])
    max_y = np.max(y_2d.flatten()[0:ny])

    # Check for row- or column-wise i.e. alignment in flat array
    if x_2d[0, 0] == min_x and x_2d[0, -1] == max_x:
        flag_transpose = 1
        flag_switch_nx_ny = 0
    # elif x_2d[0, 0] == np.min(dictionary['x'][0:nx]) and x_2d[-1, 0] == np.max(dictionary['x'][0:nx]): Undefined case
    #     flag_transpose = 0
    #     flag_switch_nx_ny = 0
    elif y_2d[0, 0] == min_y and y_2d[0, -1] == max_y and min_y != max_y:  # Correctly aligned
        flag_transpose = 0
        flag_switch_nx_ny = 0
    else:
        flag_switch_nx_ny = 1
        x_2d = dictionary['x'].reshape(ny, nx)

        # Ensure x increases in row-wise direction
        if not np.max(x_2d[:, 0]) > np.max(x_2d[0, :]):
            flag_transpose = 1

    # Apply changes to all data
    if flag_transpose and not flag_switch_nx_ny:
        adjust_orientation(dictionary, nx, ny)
    elif flag_transpose and flag_switch_nx_ny:
        adjust_orientation(dictionary, ny, nx)
    elif not flag_transpose and flag_switch_nx_ny:
        sys.exit('ERROR in check_orientation:'
                 'Cannot handle orientation in dictionary '
                 'with shape [nx:' + str(dictionary['nx']) + ',ny:' + str(dictionary['ny']) + ']')
    else:
        pass
        # print('Matrix orientation correct.')

    return dictionary


def adjust_orientation(dictionary, nx, ny):
    """
    Adjust the orientation of flat data in dictionary.
    :param ny: Number of points in x-direction.
    :param nx: Number of points in x-direction.
    :param dictionary: Dictionary of flow data for one block.
    :return: 1: success.
    """

    for dict_key in dictionary.keys():
        if isinstance(dictionary[dict_key], np.ndarray):
            dictionary[dict_key] = dictionary[dict_key].reshape(nx, ny).T.flatten()

    return 1


def remove_ghost_cells(blockdict, ymin, ymax, num_ghost_x=1):
    """
    Remove the ghost cells from each field in blockdict.
    Assume ghost cells at outer-most x-coordinates and at the
    block-internal y-boundaries, but not at walls.
    :param num_ghost_x: Number of ghost cells to remove
    along x-coordinates on one side. Symmetric boundaries assumed.
    :param ymax: Global minimum in y-coordinate for the flow case.
    :param ymin: Global minimum in y-coordinate for the flow case.
    :param blockdict: Dictionary of flow data for one block.
    :return: 1: success.
    """

    # Check whether any ghost cells need removing
    if num_ghost_x == None or num_ghost_x == 0:
        # print('No ghost cells have been removed')
        return 1

    assert(num_ghost_x > 0), "Invalid number of ghost cells: %r" % num_ghost_x

    old_nx = blockdict['nx']
    old_ny = blockdict['ny']

    flag_bottom_block = 0
    flag_top_block = 0
    flag_internal_block = 0

    # Evaluate block location
    if ymax == np.max(blockdict['y']):
        flag_top_block = 1
        new_nx = old_nx - 2*num_ghost_x
        new_ny = old_ny - 1
    elif ymin == np.min(blockdict['y']):
        flag_bottom_block = 1
        new_nx = old_nx - 2*num_ghost_x
        new_ny = old_ny - 1
    else:
        flag_internal_block = 1
        new_nx = old_nx - 2*num_ghost_x
        new_ny = old_ny - 2

    # Remove cells from each field of the block
    for key in blockdict.keys():  # Loop data in each block
        if isinstance(blockdict[key], np.ndarray):  # Ignore non-data entries
            field = blockdict[key]

            # Distinguish 2nd order tensor, vector and scalar quantity
            # Only copy n paste solution :(
            shape = field.shape[0:-1]
            # Scalar
            if shape == ():
                field = field.reshape(old_nx, old_ny)
                # Top-most block
                if flag_top_block:
                    field = field[num_ghost_x:-num_ghost_x, 1:old_ny]
                # Bottom-most block
                elif flag_bottom_block:
                    field = field[num_ghost_x:-num_ghost_x, 0:-1]
                # Internal block
                elif flag_internal_block:
                    field = field[num_ghost_x:-num_ghost_x, 1:-1]
                else:
                    sys.exit('ERROR in remove_ghost_cells:'
                             'Cannot recognise location of the block grid')
                blockdict[key] = np.array(field).flatten()

            # Vector
            elif shape == (3,):
                old_field = field.reshape(*shape, old_nx, old_ny)
                new_field = np.zeros((*shape, new_nx*new_ny))
                # Top-most block
                for i in range(shape[0]):
                    if flag_top_block:
                        field_element = old_field[i, num_ghost_x:-num_ghost_x, 1:old_ny]
                    # Bottom-most block
                    elif flag_bottom_block:
                        field_element = old_field[i, num_ghost_x:-num_ghost_x, 0:-1]
                    # Internal block
                    elif flag_internal_block:
                        field_element = old_field[i, num_ghost_x:-num_ghost_x, 1:-1]
                    else:
                        sys.exit('ERROR in remove_ghost_cells:'
                                 'Cannot recognise location of the block grid')
                    new_field[i] = np.array(field_element).flatten()
                blockdict[key] = new_field

            # 2nd order tensor
            elif shape == (3, 3):
                old_field = field.reshape(*shape, old_nx, old_ny)
                new_field = np.zeros((*shape, new_nx*new_ny))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        # Top-most block
                        if flag_top_block:
                            field_element = old_field[i, j, num_ghost_x:-num_ghost_x, 1:old_ny]
                        # Bottom-most block
                        elif flag_bottom_block:
                            field_element = old_field[i, j, num_ghost_x:-num_ghost_x, 0:-1]
                        # Internal block
                        elif flag_internal_block:
                            field_element = old_field[i, j, num_ghost_x:-num_ghost_x, 1:-1]
                        else:
                            sys.exit('ERROR in remove_ghost_cells:'
                                     'Cannot recognise location of the block grid')
                        new_field[i, j] = np.array(field_element).flatten()
                blockdict[key] = new_field
            else:
                print('ERROR in remove_ghost_cells:'
                      'Cannot handle shape ' + str(shape) + ' of field ' + str(key))
                sys.exit()

    # Correct number of points in x and y
    blockdict['nx'] = new_nx
    blockdict['ny'] = new_ny

    return 1


def merge_data(dictionary, block_arrangement):
    """
    Merge individual block-structured data into common
    dictionary for a data set.
    :param dictionary: Dictionary with blockdicts for each block's data.
    :param block_arrangement: Keys ordered according to location of blocks.
    :return: Dictionary of arrays for each data field.
    """

    # If single block, return dictionary
    if block_arrangement is None or len(block_arrangement) <= 1:
        return dictionary['block0']

    mergedict = dict()
    block_names = ['block' + str(i) for i in block_arrangement]
    for blockdict_name in block_names:
        blockdict = dictionary[blockdict_name]

        for key in blockdict.keys():
            # Check equal length of nx for all blocks
            if key == 'nx':
                if blockdict['nx'] == dictionary['block0']['nx']:
                    mergedict['nx'] = np.array([blockdict['nx']])
                    continue
                else:
                    sys.exit('ERROR in merge_data: Blocks have unequal nx. ' +
                             blockdict_name + ' has ' + str(blockdict['nx']) +
                             ' instead of ' + str(dictionary['block0']['nx']))

            # Initialisation for subsequent concatenations
            if blockdict_name == block_names[0]:
                # Transform ny to ndarray, fields are simply added to mergedict
                if key == 'ny':
                    mergedict['ny'] = np.array([blockdict['ny']])
                else:
                    mergedict[key] = blockdict[key]

            # Concatenate data
            else:
                if isinstance(blockdict[key], int):  # Skip integers
                    pass
                elif isinstance(blockdict[key], str):  # Only for dummy object
                    mergedict[key] = blockdict[key]
                    mergedict['ny'] += blockdict['ny']
                else:
                    shape_string = get_shape(blockdict[key])
                    mergedict[key] = merger(mergedict, blockdict, key, shape_string)

                    # Adjust ny after last merge
                    if key == list(mergedict)[-1]:
                        mergedict['ny'] += blockdict['ny']

    # Clean up mergedict after merge
    mergedict['nx'] = int(mergedict['nx'][0])  # Convert to int
    mergedict['ny'] = int(mergedict['ny'][0])
    mergedict['num_of_points'] = mergedict['nx']*mergedict['ny']  # Update

    return mergedict


def merger(dict1, dict2, dict_key, shape_name):
    """
    Merge arrays according to dict_key in two dictionaries.
    :param shape_name: Name of the shape (return of get_shape)
    :param dict1: Dictionary of block data.
    :param dict2: See dict1.
    :param dict_key: Data to be merged.
    :return: Merged numpy array from dict1 and dict2.
    """

    # For convenience
    nx_merge = int(dict1['nx'])
    ny_merge = int(dict1['ny'])
    nx_block = int(dict2['nx'])
    ny_block = int(dict2['ny'])

    # Get current dict_key data
    array1 = dict1[dict_key]
    array2 = dict2[dict_key]


    if shape_name == 'scalar_array':
        # Reshape into matrix
        array1 = array1.reshape(nx_merge, ny_merge)
        array2 = array2.reshape(nx_block, ny_block)

        # Concatenate matrices along columns (y-axis)
        new_array = np.concatenate((array1, array2), axis=1)
        new_array = new_array.flatten()

    elif shape_name == 'vector_array':
        new_array = np.zeros((3, nx_merge, ny_merge + ny_block))
        for i in range(3):
            comp_array1 = array1[i].reshape(nx_merge, ny_merge)
            comp_array2 = array2[i].reshape(nx_block, ny_block)
            new_array[i] = np.concatenate((comp_array1, comp_array2), axis=1)
        new_array = new_array.reshape(3, -1)

    elif shape_name == 'tensor_array':
        new_array = np.zeros((3, 3, nx_merge, ny_merge + ny_block))
        for i in range(3):
            for j in range(3):
                comp_array1 = array1[i, j].reshape(nx_merge, ny_merge)
                comp_array2 = array2[i, j].reshape(nx_block, ny_block)
                new_array[i, j] = np.concatenate((comp_array1, comp_array2), axis=1)
        new_array = new_array.reshape(3, 3, -1)

    else:
        assert False, 'Cannot handle shape: %r' % (shape_name)

    return new_array


def rename_kth_keys(dictionary):
    """
    Rename keys from kth style to defined convention.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """
    convention_keys = ['um', 'vm', 'wm',
                       'pm', 'p_rms',
                       'prod_k', 'diss_rt', 'trb_tsp_k',
                       'vsc_dif_k', 'vel_prs_grd_k', 'conv_k',
                       'u_tau', 'Re_tau']
    kth_keys = ['U', 'V', 'W',
                'P', 'prms',
                'Pk', 'Dk', 'Tk',
                'VDk', 'VPGk', 'Ck',
                'ut', 'Ret']

    for convention_key, kth_key in zip(convention_keys, kth_keys):
        try:
            dictionary[convention_key] = dictionary.pop(kth_key)
        except KeyError:
            pass

    return 1


def rename_subdicts(dictionary, subcase_names):
    """
    Rename the subdicts according to subcase_names. Assume
    order is identical.
    :param dictionary: Dictionary as loaded from kth mat files with
    only subdicts of distinct flow cases.
    :param subcase_names: Names of individual subcases.
    :return: 1:success.
    """

    # Rename each dictionary
    for subdict_key, subcase_name in zip(tuple(dictionary), subcase_names):
        dictionary[subcase_name] = dictionary[subdict_key]
        del dictionary[subdict_key]

    return 1


def check_tauij_sign(dictionary):
    """
    Loop all components of the reynolds stress tensor and set numerical
    artifacts to zero.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """

    flag_change = 0

    # Loop all possible tauij elements
    tau_keys = TAU_KEYS
    for tau_key in tau_keys:
        if tau_key in dictionary:
            for i, value in enumerate(dictionary[tau_key].flatten()):
                # Detect numerical artifacts as negative and very small quantities
                if value < 1e-10:
                    dictionary[tau_key].ravel()[i] = 0
                    flag_change = 1

    if flag_change:
        # print('Found and corrected numerical artifacts in tauij')
        pass

    return 1


def get_nx_ny(dictionary, nx=0, ny=0, nz=0):
    """
    Find number of points in x and y direction.
    :param dictionary: Dictionary of raw flow data.
    :return: 1:success.
    """

    assert any([coord in dictionary for coord in ['x', 'y', 'z']]), 'No geometry data available. ' \
                                                                    'Neither x, y, or z key found in dictionary.'
    if all(num in dictionary for num in ['nx', 'ny']):
        return 1

    if all([nx, ny, nz]):
        dictionary['nx'] = nx
        dictionary['ny'] = ny
        dictionary['nz'] = nz
    elif len(dictionary['y'].shape) > 1:
        dictionary['nx'] = dictionary['y'].shape[0]
        dictionary['ny'] = dictionary['y'].shape[1]
    else:
        assert False, 'Variable dimension does not allow reading nx and ny, invalid shape is: %r' % (dictionary['y'].shape)

    return 1


def set_char_length(dictionary, char_length):
    """
    Set the characteristic length for each coordinate.
    Use if length is not uniform.
    :param char_length: Scalar or key for the length scale.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """

    # Check definition of characteristic length parameter
    if isinstance(char_length, (int, float, np.ndarray)):
        dictionary['char_length'] = char_length
        return 1

    elif isinstance(char_length, str):
        assert char_length in dictionary, 'Invalid key for characteristic length: %r' % char_length
        # For convenience
        nx = dictionary['nx']
        ny = dictionary['ny']

        length = np.zeros((nx, ny))
        for i in range(nx):
            length[i, :] = dictionary[char_length][i]  # Characteristic length varies with x
        dictionary['char_length'] = length

    else:
        assert False, 'Unknown type for characteristic length key: char_length'

    return 1


def set_offset_from_wall(dictionary, wall_offset):
    """
    Set the offset from the wall either as scalar or as an array of
    x-dependent length.
    :param dictionary: Dictionary of flow data.
    :param wall_offset: Scalar or key in dictionary.
    :return: 1:success.
    """

    # Check definition of wall offset parameter
    if isinstance(wall_offset, (int, float, np.ndarray)):
        dictionary['wall_offset'] = wall_offset
        return 1

    elif isinstance(wall_offset, str):
        assert wall_offset in dictionary, 'Invalid key for characteristic length: %r' % wall_offset
        # For convenience
        nx = dictionary['nx']
        ny = dictionary['ny']

        offset = np.zeros((nx, ny))
        for i in range(nx):
            offset[i, :] = dictionary[wall_offset][i]  # Wall offset varies with x
        dictionary['wall_offset'] = offset

    else:
        assert False, "Cannot handle type for wall offset key: %r" % (type(wall_offset))



def generate_2d_coord(dictionary, transpose=False):
    """
    Generate full x- and y-coordinates for each point with given
    vector for rectilinear coordinates.
    :param transpose: Transpose coordinate array to fit structure of
    given data, choose manually.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """

    # For convenience
    nx = dictionary['nx']
    ny = dictionary['ny']
    num_of_points = nx*ny

    # Generate full x-coordinates
    if dictionary['x'].ravel().shape[0] == num_of_points:
        pass
    else:
        # Generate coordinates in correct structure
        x_2d = np.zeros((nx, ny))
        for i in range(nx):
            x_2d[i, :] = dictionary['x'][i]
        # Fit structure of coordinates to other given data
        if transpose:
            dictionary['x'] = x_2d.T.flatten()
        else:
            dictionary['x'] = x_2d.flatten()

    # Generate full y-coordinates
    if dictionary['y'].ravel().shape[0] == num_of_points:
        pass
    else:
        # Generate coordinates in correct structure
        y_2d = np.zeros((nx, ny))
        for i in range(ny):
            y_2d[:, i] = dictionary['y'][i]

        # Fit structure of coordinates to other given data
        if transpose:
            dictionary['y'] = y_2d.T.flatten()
        else:
            dictionary['y'] = y_2d.flatten()


    return 1


def flatten_all_arrays(dictionary):
    """
    Apply flatten function to each ndarray in dictionary.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """

    for var_key in dictionary:
        if isinstance(dictionary[var_key], np.ndarray):
            dictionary[var_key] = dictionary[var_key].flatten()
        else:
            pass

    return 1


def get_polar_coordinates(dictionary, coordinate_system):
    """
    Compute coordinates and near-wall quantities for pipe flow.
    :param coordinate_system: Specification of coordinate system.
    :param dictionary: Dictionary of flow data.
    :return: 1: success.
    """

    # No cylinder coordinates necessary for cartesian data
    if coordinate_system == 'cartesian':
        return 1

    # For convenience
    x = dictionary['x']
    y = dictionary['y']

    # Compute polar coordinates
    dictionary['r'] = np.sqrt(x**2 + y**2)  # Radius
    dictionary['t'] = np.arctan(safe_divide(y, x))  # Theta

    # Set number of cells in angular and radial directions
    dictionary['nt'] = dictionary['nx']
    dictionary['nr'] = dictionary['ny']

    return 1


def remove_central_point(dictionary, coordinate_system):
    """
    A common central point in cylinder coordinates causes a singular
    jacobi matrix for gradients and transformations. Hence, remove it.
    :param coordinate_system: Specification of coordinate system.
    :param dictionary: Dictionary of flow data.
    :return: 1:success.
    """

    # No central point in cartesian coordinates
    if coordinate_system == 'cartesian':
        return 1

    # For convenience
    nt = dictionary['nt']
    nr = dictionary['nr']

    x = dictionary['x'].reshape(nt, nr)
    y = dictionary['y'].reshape(nt, nr)

    # Check for common central point
    if x[:, 0].any() == 0 and y[:, 0].any() == 0:
        # Remove each first row i.e. the central point
        for dict_key in dictionary:
            element = dictionary[dict_key]
            if isinstance(element, np.ndarray):
                element = element.reshape(nt, nr)
                element = element[:, 1:]
                dictionary[dict_key] = element.flatten()
            elif dict_key == 'ny' or dict_key == 'nr':
                dictionary[dict_key] = element - 1
            else:
                pass


    return 1


def find_solid_grid_points(dictionary, geo):
    """
    Find points inside the solid geometry for a the non-dimensional
    periodic hills geometry.
    :param geo: geometry key in subdict, required for information
    about geometry.
    :param dictionary: Dictionary of flow data.
    :return: Index for each point in flat array.
    """

    # Check scaling factor of the geometry
    if 'geometry_scale' in dictionary:
        geo_scale = dictionary['geometry_scale']
    else:
        geo_scale = 1

        # Find points inside the solid geometry by index
    idx = list()
    for x_point, y_point, i in zip(dictionary['x'], dictionary['y'], np.arange(dictionary['nx']*dictionary['ny'])):
        if geo == 'periodic_hills':
            y_loc_boundary = geometry_periodic_hills_lower(x_point, geo_scale)
        else:
            y_loc_boundary = -np.inf

        if y_point < y_loc_boundary:
            idx.append(i)

    return idx


def set_solid_grid_points_to_zero(dictionary, geo):
    """
    Set all values (except coordinates) to zero, if point
    is outside of the geometry.
    :param dictionary: Dictionary of flow data.
    :return: 1:success.
    """

    # Get index of solid grid points
    idx = find_solid_grid_points(dictionary, geo)

    # Set each dictionary values of solid points to zero
    for i in idx:
        for dic_key in tuple(dictionary):
            # Skip coordinates
            if dic_key == 'x' or dic_key == 'y' or dic_key == 'z':
                continue
            variable = dictionary[dic_key]
            if isinstance(variable, np.ndarray):
                shape = get_shape(variable)
                if shape == 'scalar_array':
                    variable[i] = 0
                elif shape == 'vector_array':
                    variable[:, i] = 0
                elif shape == 'tensor_array':
                    variable[:, :, i] = 0

    return 1


