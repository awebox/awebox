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
"""
Class Architecture contains functions to facilitate architecture construction
and information storage and retrieval.

@author: jochem de schutter alu-freiburg 2018,
@edit: rachel leuthold, alu-fr 2019
"""


class Architecture:
    """Class that facilitates structure bookkeeping of tree-structured tethered kite systems.
    """
    def __init__(self, parent_map):

        self.__parent_map = parent_map
        self.__number_of_nodes = len(list(parent_map.keys()))+1
        self.__build_kite_nodes()
        self.__build_layers()
        self.__build_children_map()
        self.__build_siblings_map()
        self.__build_kites_map()

        return None

    def __build_kite_nodes(self):
        """Construct a list containing all the node numbers that have kite attached to them
        """

        kite_nodes = []
        for node in list(self.__parent_map.keys()):
            if node not in list(self.__parent_map.values()):
                kite_nodes += [node]

        self.__kite_nodes = kite_nodes
        self.__number_of_kites = len(kite_nodes)

        return None

    def __build_layers(self):
        """Get number of layers in the tree and build siblings dict (number of siblings per layer)
        """

        if self.__number_of_nodes == 2: # single kites

            self.__layer_nodes = [0]
            self.__layers = 1

        else: # multiple kites
            self.__layer_nodes = list(set(self.__parent_map.values())-set([0]))
            self.__layers = len(self.__layer_nodes)

        return None

    def __build_children_map(self):
        """Build inverse parent map
        """
        if self.__number_of_nodes == 2: # single kites

            self.__children_map = {0:[1]}
            self.__children = {0:1}

        else: # multiple kites

            parents = list(set(self.__parent_map.values()))
            children_map = {}
            children = {}
            for parent in parents:
                children_map[parent] = [k for k,v in self.__parent_map.items() if (v == parent)]
                children[parent] = len(children_map[parent])
            self.__children = children
            self.__children_map = children_map

        return None

    def __build_kites_map(self):
        """Get list of kite nodes in the children map of a node
        """

        parents = list(set(self.__parent_map.values()))

        kites_map = {}
        for parent in parents:
            kites_map[parent] = list(filter(lambda x: x in self.__kite_nodes, self.__children_map[parent]))
        self.__kites_map = kites_map

        return None

    def __build_siblings_map(self):
        """ Build siblings map. Maps from one kite to list of all kites in the same layer
        including itself.
        """

        siblings_map = {}
        for kite in self.__kite_nodes:
            siblings_map[kite] = list(filter(lambda x: x in self.__kite_nodes,
                self.__children_map[self.__parent_map[kite]]))
        self.__siblings_map = siblings_map

        return None

    def get_number_children(self, parent):
        children = self.__children_map[parent]
        number_children = len(children)
        return number_children

    def get_number_siblings(self, kite):
        siblings = self.__siblings_map[kite]
        number_siblings = len(siblings)
        return number_siblings

    def get_all_level_siblings(self):

        parent_map = self.__parent_map
        kite_nodes = self.__kite_nodes

        level_siblings = {}
        for kite in kite_nodes:
            parent = parent_map[kite]

            if not (parent in list(level_siblings.keys())):
                level_siblings[parent] = []

            level_siblings[parent] += [kite]

        return level_siblings

    def node_label(self, node):
        return str(node) + str(self.__parent_map[node])

    def parent_label(self, node):

        parent = self.__parent_map[node]
        if node > 1:
            grandparent = self.__parent_map[parent]
        else:
            grandparent = 0

        return str(parent) + str(grandparent)
        
    @property
    def parent_map(self):
        """parent node map"""
        return self.__parent_map

    @parent_map.setter
    def parent_map(self, value):
        print('Cannot set parent_map object.')

    @property
    def number_of_nodes(self):
        """number of nodes in tree structure"""
        return self.__number_of_nodes

    @number_of_nodes.setter
    def number_of_nodes(self, value):
        print('Cannot set number_of_nodes object.')

    @property
    def kite_nodes(self):
        """list of kite nodes"""
        return self.__kite_nodes

    @kite_nodes.setter
    def kite_nodes(self, value):
        print('Cannot set kite_nodes object.')

    @property
    def number_of_kites(self):
        """number of kites in the tree"""
        return self.__number_of_kites

    @number_of_kites.setter
    def number_of_kites(self, value):
        print('Cannot set number_of_kites object.')

    @property
    def layers(self):
        """number of layers in the tree"""
        return self.__layers

    @layers.setter
    def layers(self, value):
        print('Cannot set layers object.')

    @property
    def layer_nodes(self):
        """number of layer nodes in the tree"""
        return self.__layer_nodes

    @layer_nodes.setter
    def layer_nodes(self, value):
        print('Cannot set layer_nodes object.')

    @property
    def children_map(self):
        """map of all children, sorted by parent nodes"""
        return self.__children_map

    @children_map.setter
    def children_map(self, value):
        print('Cannot set children_map object.')

    @property
    def children(self):
        """number of children per parent in the tree"""
        return self.__children

    @children.setter
    def children(self, value):
        print('Cannot set children object.')

    @property
    def kites_map(self):
        """map of all kite children, sorted by parent nodes"""
        return self.__kites_map

    @kites_map.setter
    def kites_map(self, value):
        print('Cannot set kites_map object.')

    @property
    def siblings_map(self):
        """sibling nodes sorted by layer in the tree"""
        return self.__siblings_map

    @siblings_map.setter
    def siblings_map(self, value):
        print('Cannot set siblings_map object.')

    @property
    def siblings(self):
        """number of siblings per layer in the tree"""
        return self.__siblings

    @siblings.setter
    def siblings(self, value):
        print('Cannot set siblings object.')
