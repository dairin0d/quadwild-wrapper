'''
Copyright (C) 2015-2024 Team C All Rights Reserved

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

bl_info = {
    'name': 'QuadWild wrapper',
    'description': 'Remeshing based on the QuadWild library',
    'author': 'TeamC, dairin0d',
    'version': (1, 0, 0),
    'blender': (3, 6, 0),
    'location': 'View3D',
    'wiki_url': '',
    'category': 'Mesh',
}

from . import src


def register():
    src.register()


def unregister():
    src.unregister()
