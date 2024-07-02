import sys
import os
import subprocess
import math
import time

import bpy
import bmesh
from mathutils import Vector

import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .rosy_combing import (
    RoSyOptions,
    RETOPO_OT_rosy_mouse_detect,
    RETOPO_OT_rosy_reset,
    RETOPO_OT_rosy_relax,
    RETOPO_OT_rosy_combing,
    RoSy,
)

from .priority_heap import PriorityHeap


class QuadWildConfig(bpy.types.PropertyGroup):
    is_installed: bpy.props.BoolProperty(
        name='Is QuadWild installed?',
        description='',
        get=(lambda self: QuadWild.is_installed))

    detail_mode: bpy.props.EnumProperty(
        name='Detail mode',
        description='Method to control the tesselation detail',
        items=[
            ('DETAIL', 'Detail', 'Tweak the detail level (relative to the default)'),
            ('POLYCOUNT', 'Polycount', 'Specify the approximate target number of polygons'),
        ],
        default='DETAIL')

    detail: bpy.props.FloatProperty(
        name='Detail',
        description='Higher values -> more polygons, lower values -> fewer polygons',
        min=0.01,
        step=1,
        default=1)

    polycount: bpy.props.IntProperty(
        name='Polycount',
        description='Target number of polygons (NOTE: mesh will be processed multiple times)',
        min=1,
        default=10000)

    use_base_mesh: bpy.props.BoolProperty(
        name='Use base mesh',
        description='Use the base mesh (without modifiers)',
        default=False)

    sharp_conditions: bpy.props.EnumProperty(
        name='Preserve edges',
        description='Which edges should be preserved during retopology',
        options={'ENUM_FLAG'},
        items=[
            ('NORMAL', 'Normal', 'Edges with mismatching normals on different sides'),
            ('ANGLE', 'Angle', 'Edges (and vertices) with angle above the specified limit'),
            ('SHARP', 'Sharp', 'Edges marked as Sharp'),
            ('SEAM', 'Seam', 'Edges marked as Seam'),
            ('MATERIAL', 'Material', 'Edges between faces with different materials'),
            ('UV', 'UVs', 'Edges with mismatching UV coordinates on different sides'),
        ],
        default={'ANGLE', 'SHARP', 'SEAM', 'MATERIAL', 'UV'})

    normals_angle: bpy.props.FloatProperty(
        name='Normals angle',
        description='The angle between normals at which they are considered discontinuous',
        subtype='ANGLE',
        min=0,
        max=math.pi,
        step=1,
        default=math.radians(0.01))

    sharp_angle: bpy.props.FloatProperty(
        name='Sharp angle',
        description='The angle for automatic determination of sharp edges',
        subtype='ANGLE',
        min=0,
        max=math.pi,
        step=100,
        default=math.radians(35))

    use_preprocess: bpy.props.BoolProperty(
        name='Preprocess',
        description='Enable mesh preprocessing',
        default=True)

    use_preprocess_simplify: bpy.props.BoolProperty(
        name='Simplify',
        description='Merge nearly-coplanar polygons to reduce the processing time',
        default=True)

    planar_angle: bpy.props.FloatProperty(
        name='Planar angle',
        description='Angle between faces at which they are considered coplanar and can be simplified',
        subtype='ANGLE',
        min=0,
        max=math.pi,
        step=100,
        default=math.radians(5))

    use_preprocess_subdivide: bpy.props.BoolProperty(
        name='Subdivide',
        description='Perform extra mesh subdivision(s)',
        default=True)

    subdivide_iterations: bpy.props.IntProperty(
        name='Subdivision iterations',
        description='QuadWild may fail on some lowpoly meshes / simple polygons',
        min=0,
        max=4,
        default=2)

    quad_regularity: bpy.props.FloatProperty(
        name='Quad regularity',
        description='Regularity of quads in the retopology stage',
        subtype='FACTOR',
        min=0,
        max=1,
        precision=3,
        step=1,
        default=0.995)

    non_quad_regularity: bpy.props.FloatProperty(
        name='Non-quad regularity',
        description='Regularity of non-quads in the retopology stage',
        subtype='FACTOR',
        min=0,
        max=1,
        precision=3,
        step=1,
        default=0.9)

    ilp_method: bpy.props.EnumProperty(
        name='ILP method',
        description='Solver method for Integer Linear Programming',
        items=[
            ('LEASTSQUARES', 'Least Squares', ''),
            ('ABS', 'Absolute', ''),
        ],
        default='ABS')

    full_solve: bpy.props.BoolProperty(
        name='Full solve',
        description='Attempt a full solve, no matter how much time it takes',
        default=False)

    singularity_alignment: bpy.props.FloatProperty(
        name='Singularity alignment',
        description='Singularity alignment weight',
        subtype='FACTOR',
        min=0,
        max=1,
        precision=3,
        step=1,
        default=0.1)

    lost_constraint_iterations: bpy.props.IntProperty(
        name='Lost constraint iterations',
        description='',
        min=0,
        default=1)

    lost_constraint_flags: bpy.props.EnumProperty(
        name='Lost constraint options',
        description='',
        options={'ENUM_FLAG'},
        items=[
            ('QUADS', 'Quads', ''),
            ('NON_QUADS', 'Non-quads', ''),
            ('ALIGN', 'Align', ''),
        ],
        default={'ALIGN'})

    hard_parity_constraint: bpy.props.BoolProperty(
        name='Hard parity constraint',
        description='',
        default=True)

    fixed_chart_clusters: bpy.props.IntProperty(
        name='Fixed chart clusters',
        description='',
        min=0,
        default=0)

    use_flow_solver: bpy.props.BoolProperty(
        name='Use flow solver',
        description='',
        default=True)

    flow_config: bpy.props.EnumProperty(
        name='Flow config',
        description='',
        items=[
            ('VIRTUAL_SIMPLE', 'Virtual Simple', ''),
            ('VIRTUAL_HALF', 'Virtual Half', ''),
        ],
        default='VIRTUAL_SIMPLE')

    flow_solver_config: bpy.props.EnumProperty(
        name='Flow solver config',
        description='',
        items=[
            ('DEFAULT', 'Default', ''),
            ('LEMON', 'Lemon', ''),
            ('EDGETHRU', 'Edgethru', ''),
            ('NODETHRU', 'Nodethru', ''),
            ('APPROX_MST', 'Approx. MST', ''),
            ('APPROX_ROUND2EVEN', 'Approx. Round2Even', ''),
            ('APPROX_SYMMDC', 'Approx. SymMDC', ''),
        ],
        default='DEFAULT')

    def draw_installation_ui(self, layout):
        if QuadWild.is_installed:
            layout.label(text='QuadWild is installed', icon='CHECKMARK')
        else:
            layout.operator('retopo.quadwild_install', icon='IMPORT')
        
        if QuadWild.installation_error:
            layout.label(text=QuadWild.installation_error, icon='ERROR')


class types():
    class scene(bpy.types.PropertyGroup):
        rosy: bpy.props.PointerProperty(type=RoSyOptions)
        config: bpy.props.PointerProperty(type=QuadWildConfig)


def get_os_type():
    if sys.platform.startswith('linux'):
        return 'LINUX'
    elif sys.platform.startswith('win32') or sys.platform.startswith('cygwin'):
        return 'WINDOWS'
    elif sys.platform.startswith('darwin'):
        return 'MACOS'
    else:
        return 'UNSUPPORTED'


def macos_unquarantine(path):
    if os.path.isdir(path):
        path = os.path.join(path, '*')
    
    # This is a command recommended by the author of quadwild bi-mdf:
    subprocess.run(['xattr', '-d', 'com.apple.quarantine', path])


class QuadWild:
    bin_path = ''
    prep_path = ''
    main_path = ''

    tmp_path = ''
    mesh_names = [
        'mesh.obj',
        'mesh_rem.obj',
        'mesh_rem_p0.obj',
        'mesh_rem_p0_1_quadrangulation.obj',
        'mesh_rem_p0_1_quadrangulation_smooth.obj',
    ]

    platform_infos = {
        'LINUX': {
            'url': 'https://github.com/cgg-bern/quadwild-bimdf/releases/download/v0.0.2/linux-binaries.zip',
            'zip': 'linux-binaries.zip',
            'unquarantine': None,
            'prep': 'quadwild',
            'main': 'quad_from_patches',
        },
        'WINDOWS': {
            'url': 'https://github.com/cgg-bern/quadwild-bimdf/releases/download/v0.0.2/windows-binaries.zip',
            'zip': 'windows-binaries.zip',
            'unquarantine': None,
            'prep': 'quadwild.exe',
            'main': 'quad_from_patches.exe',
        },
        'MACOS': {
            'url': 'https://github.com/cgg-bern/quadwild-bimdf/releases/download/v0.0.2/macos-binaries.zip',
            'zip': 'macos-binaries.zip',
            'unquarantine': macos_unquarantine,
            'prep': 'quadwild',
            'main': 'quad_from_patches',
        },
    }

    platform_info = None

    is_os_supported = False
    is_installed = False
    installation_error = ''

    flow_configs = {
        'VIRTUAL_SIMPLE': 'config/main_config/flow_virtual_simple.json',
        'VIRTUAL_HALF': 'config/main_config/flow_virtual_half.json',
    }

    flow_solver_configs = {
        'DEFAULT': 'config/satsuma/default.json',
        'LEMON': 'config/satsuma/lemon.json',
        'EDGETHRU': 'config/satsuma/edgethru.json',
        'NODETHRU': 'config/satsuma/nodethru.json',
        'APPROX_MST': 'config/satsuma/approx-mst.json',
        'APPROX_ROUND2EVEN': 'config/satsuma/approx-round2even.json',
        'APPROX_SYMMDC': 'config/satsuma/approx-symmdc.json',
    }

    object_types = {
        'MESH',
        'CURVE',
        'SURFACE',
        'META',
        'FONT',
    }

    @classmethod
    def initialize(cls):
        # Get rid of symlinks, just in case
        module_path = os.path.realpath(os.path.dirname(__file__))
        
        cls.bin_path = os.path.join(module_path, 'bin')
        cls.tmp_path = os.path.join(bpy.app.tempdir, 'quadwild')
        
        os_type = get_os_type()
        cls.platform_info = cls.platform_infos.get(os_type)
        
        cls.is_os_supported = bool(cls.platform_info)
        if not cls.is_os_supported:
            cls.installation_error = 'OS platform is not supported'
            return
        
        cls.prep_path = os.path.join(cls.bin_path, cls.platform_info['prep'])
        cls.main_path = os.path.join(cls.bin_path, cls.platform_info['main'])
        
        cls.is_installed = os.path.isfile(cls.prep_path) and os.path.isfile(cls.main_path)
        
        if not cls.is_installed:
            cls.installation_error = 'QuadWild is not installed'

    @classmethod
    def install(cls):
        if not cls.platform_info:
            cls.installation_error = 'OS platform is not supported'
            return
        
        cls.installation_error = ''
        
        from urllib.request import Request, urlopen
        from urllib.error import URLError
        import shutil
        import tempfile
        
        req = Request(cls.platform_info['url'])
        
        try:
            response = urlopen(req)
        except URLError as e:
            if hasattr(e, 'reason'):
                cls.installation_error = f'Failed to reach a server. Reason: {e.reason}'
            elif hasattr(e, 'code'):
                cls.installation_error = f'The server could not fulfill the request. Error code: {e.code}'
            return
        
        try:
            with response:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    shutil.copyfileobj(response, tmp_file)
        except Exception as exc:
            cls.installation_error = f'Could not download the file. {exc}'
            return
        
        cls.unpack_binaries(tmp_file.name)
        
        cls.is_installed = os.path.isfile(cls.prep_path) and os.path.isfile(cls.main_path)
        
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

    @classmethod
    def unpack_binaries(cls, archive_path):
        if not os.path.isfile(archive_path):
            cls.installation_error = f'Could not find {archive_path}'
            return
        
        import shutil
        import zipfile
        
        try:
            if os.path.exists(cls.bin_path):
                shutil.rmtree(cls.bin_path, ignore_errors=True)
        except Exception as exc:
            cls.installation_error = f'Could not clear the installation directory. {exc}'
            return
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(cls.bin_path)
        except Exception as exc:
            cls.installation_error = f'Could not extract archive. {exc}'
            return
        
        unquarantine = cls.platform_info['unquarantine']
        
        if unquarantine:
            try:
                unquarantine(cls.bin_path)
            except Exception as exc:
                cls.installation_error = f'Could not unquarantine quadwild. {exc}'

    @classmethod
    def last_mesh_path(cls, i_min=0, i_max=-1):
        if i_min < 0: i_min += len(cls.mesh_names)
        if i_max < 0: i_max += len(cls.mesh_names)
        
        for i in range(i_max, i_min-1, -1):
            mesh_path = os.path.join(cls.tmp_path, cls.mesh_names[i])
            if os.path.exists(mesh_path):
                return i, mesh_path
        
        return -1, ''

    @classmethod
    def sanitize_config_text(cls, text):
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    @classmethod
    def make_configs(cls, config, detail=None):
        scale_factor = 1.0 / (config.detail if detail is None else detail)
        
        # Don't use QuadWild's built-in remesher (it fails
        # in some cases and ignores sharp edges on planes)
        do_remesh = False
        sharp_feature_threshold = -1
        remesh_regularity = 0.99
        
        prep_config = cls.sanitize_config_text(f'''
        do_remesh {int(do_remesh)}
        sharp_feature_thr {sharp_feature_threshold}
        alpha {1 - remesh_regularity}
        scaleFact {scale_factor}
        ''')
        
        ilp_method_id = (1 if config.ilp_method == 'ABS' else 0)
        
        if config.full_solve:
            time_limit = 86400
            gap_limit = 1.1e-9
            minimum_gap = 100
            callback_time_limit = "0"
            callback_gap_limit = "0"
        else:
            time_limit = 200
            gap_limit = 0.0
            minimum_gap = 0.4
            callback_time_limit = "8 3.0 5.0 10.0 20.0 30.0 60.0 90.0 120.0"
            callback_gap_limit = "8 0.005 0.02 0.05 0.10 0.15 0.20 0.25 0.3"
        
        lost_constraint_quads = 'QUADS' in config.lost_constraint_flags
        lost_constraint_non_quads = 'NON_QUADS' in config.lost_constraint_flags
        lost_constraint_align = 'ALIGN' in config.lost_constraint_flags
        lost_constraint_align = lost_constraint_align and (config.singularity_alignment > 0)
        
        flow_config = cls.flow_configs[config.flow_config]
        flow_solver_config = cls.flow_solver_configs[config.flow_solver_config]
        
        main_config = cls.sanitize_config_text(f'''
        alpha {1 - config.quad_regularity}
        ilpMethod {ilp_method_id}
        timeLimit {time_limit}
        gapLimit {gap_limit}
        callbackTimeLimit {callback_time_limit}
        callbackGapLimit {callback_gap_limit}
        minimumGap {minimum_gap}
        isometry {int(config.quad_regularity < 1)}
        regularityQuadrilaterals {int(config.quad_regularity > 0)}
        regularityNonQuadrilaterals {int(config.non_quad_regularity > 0)}
        regularityNonQuadrilateralsWeight {1 - config.non_quad_regularity}
        alignSingularities {int(config.singularity_alignment > 0)}
        alignSingularitiesWeight {config.singularity_alignment}
        repeatLosingConstraintsIterations {config.lost_constraint_iterations}
        repeatLosingConstraintsQuads {int(lost_constraint_quads)}
        repeatLosingConstraintsNonQuads {int(lost_constraint_non_quads)}
        repeatLosingConstraintsAlign {int(lost_constraint_align)}
        hardParityConstraint {int(config.hard_parity_constraint)}
        scaleFact {scale_factor}
        fixedChartClusters {config.fixed_chart_clusters}
        useFlowSolver {int(config.use_flow_solver)}
        flow_config_filename "{flow_config}"
        satsuma_config_filename "{flow_solver_config}"
        ''')
        
        return {'prep': prep_config, 'main': main_config}

    @classmethod
    def beautify_triangles(cls, bm, angle_limit=0.001):
        bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
        
        flip_heap = PriorityHeap()
        
        pi = math.pi
        
        def tri_angle(v0, v1, v2):
            return (v1.co - v0.co).angle(v2.co - v0.co, pi)
        
        def update_flip_cost(edge):
            if (not edge.smooth) or (edge.calc_face_angle(pi) > angle_limit):
                cost = 0
            else:
                v0, v1 = edge.verts
                v2 = edge.link_loops[0].link_loop_prev.vert
                v3 = edge.link_loops[1].link_loop_prev.vert
                angle0 = max(tri_angle(v2, v0, v1), tri_angle(v3, v1, v0))
                angle1 = max(tri_angle(v0, v2, v3), tri_angle(v1, v3, v2))
                if angle1 > 0:
                    cost = angle0 / angle1
                else:
                    cost = 0
            
            if cost > 1.001:
                flip_heap.add(edge, cost)
            elif edge in flip_heap:
                flip_heap.remove(edge)
        
        for edge in bm.edges:
            update_flip_cost(edge)
        
        while flip_heap:
            new_edge = bmesh.utils.edge_rotate(flip_heap.pop())
            update_flip_cost(new_edge)
            
            for face in new_edge.link_faces:
                for edge in face.edges:
                    if edge is not new_edge:
                        update_flip_cost(edge)

    @classmethod
    def mark_conical_apexes(cls, bm, sharp_angle):
        sharp_cos = math.cos(sharp_angle*0.5)
        
        apexes = set()
        for v in bm.verts:
            if not all(e.smooth for e in v.link_edges):
                continue
            
            normal = Vector()
            for l in v.link_loops:
                normal += l.face.normal * l.calc_angle()
            normal.normalize()
            
            if any((normal.dot(f.normal) < sharp_cos) for f in v.link_faces):
                apexes.add(v)
        
        def find_candidate_edge(v):
            for e in v.link_edges:
                if not e.smooth:
                    return None
                if e.other_vert(v) in apexes:
                    return e
            return e
        
        # To preserve conical apexes, at least one edge must be sharp
        for v in apexes:
            e = find_candidate_edge(v)
            if not e:
                continue
            e.smooth = False

    @classmethod
    def get_edge_continuity(cls, bm, corner_normals, continuity_threshold):
        edge_normals = {}
        # Note: bm.loops does not support len() or iteration
        loops = (l for f in bm.faces for l in f.loops)
        for loop, normal in zip(loops, corner_normals):
            edge_normals.setdefault((loop.edge, loop.vert), []).append(normal)
            edge2 = loop.link_loop_prev.edge
            edge_normals.setdefault((edge2, loop.vert), []).append(normal)
        
        edge_continuity = {}
        for edge in bm.edges:
            normals0 = edge_normals.get((edge, edge.verts[0]))
            normals1 = edge_normals.get((edge, edge.verts[1]))
            manifold0 = normals0 and (len(normals0) == 2)
            manifold1 = normals1 and (len(normals1) == 2)
            if manifold0 and manifold1:
                n0 = (normals0[0].dot(normals0[1]) >= continuity_threshold)
                n1 = (normals1[0].dot(normals1[1]) >= continuity_threshold)
                edge_continuity[edge] = n0 and n1
            else:
                edge_continuity[edge] = False
        
        return edge_continuity.get

    @classmethod
    def mark_sharp_edges(cls, bm, corner_normals, config):
        sharp_angle = config.sharp_angle
        continuity_threshold = math.cos(config.normals_angle)
        
        sharp_conditions = config.sharp_conditions
        
        sharp_detectors = []
        
        sharp_detectors.append(lambda edge: not edge.is_contiguous)
        
        if 'ANGLE' in sharp_conditions:
            pi = math.pi
            sharp_detectors.append(lambda edge: edge.calc_face_angle(pi) > sharp_angle)
        
        if ('NORMAL' in sharp_conditions) and corner_normals:
            edge_continuity = cls.get_edge_continuity(bm, corner_normals, continuity_threshold)
            sharp_detectors.append(lambda edge: not edge_continuity(edge, True))
        
        if 'SHARP' in sharp_conditions:
            sharp_detectors.append(lambda edge: not edge.smooth)
        
        if 'SEAM' in sharp_conditions:
            sharp_detectors.append(lambda edge: edge.seam)
        
        if 'MATERIAL' in sharp_conditions:
            def sharp_detector(edge):
                return edge.link_faces[0].material_index != edge.link_faces[1].material_index
            sharp_detectors.append(sharp_detector)
        
        if 'UV' in sharp_conditions:
            isclose = math.isclose
            def compare_uv(uvA, uvB):
                return isclose(uvA.x, uvB.x) and isclose(uvA.y, uvB.y)
            
            uv_layers = list(bm.loops.layers.uv.values())
            def sharp_detector(edge):
                loopA0, loopB1 = edge.link_loops
                loopA1 = loopA0.link_loop_next
                loopB0 = loopB1.link_loop_next
                for uv_layer in uv_layers:
                    if not compare_uv(loopA0[uv_layer].uv, loopB0[uv_layer].uv):
                        return True
                    if not compare_uv(loopA1[uv_layer].uv, loopB1[uv_layer].uv):
                        return True
                return False
            sharp_detectors.append(sharp_detector)
        
        for e in bm.edges:
            e.smooth = not any(detector(e) for detector in sharp_detectors)

    @classmethod
    def preprocess_geometry(cls, bm, corner_normals, config):
        sharp_angle = config.sharp_angle
        
        angle_limit = config.planar_angle
        
        if bm.is_wrapped:
            bm = bm.copy()
            outside_verts = [v for v in bm.verts if not any(f.select for f in v.link_faces)]
        else:
            outside_verts = [v for v in bm.verts if v.is_wire]
        
        bmesh.ops.delete(bm, geom=outside_verts, context='VERTS')
        
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.00001)
        
        cls.mark_sharp_edges(bm, corner_normals, config)
        
        if config.use_preprocess and config.use_preprocess_simplify:
            bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='BEAUTY')
            
            # Remove excessive detail so that QuadWild won't take forever on dense meshes
            bmesh.ops.dissolve_limit(bm, verts=bm.verts, edges=bm.edges, delimit={'SHARP'}, angle_limit=angle_limit)
            
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.is_wire], context='VERTS')
            
            cls.beautify_triangles(bm, angle_limit)
        else:
            cls.beautify_triangles(bm)
        
        if config.use_preprocess and config.use_preprocess_subdivide:
            # QuadWild can fail on simple polygons, but a couple of
            # subdivision iterations seem sufficient to circumvent that.
            # Plus, we probably want the input mesh to have some subdivision anyway.
            for iteration in range(config.subdivide_iterations):
                bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=1)
                cls.beautify_triangles(bm, angle_limit)
        
        cls.mark_conical_apexes(bm, sharp_angle)
        
        return bm

    @classmethod
    def copy_as_mesh(cls, src_obj, config, mesh=None, name=None, force_convert=False):
        # Currently don't know how to properly override context for these operators
        bpy.ops.object.select_all(action='DESELECT')
        src_obj.select_set(True)
        bpy.context.view_layer.objects.active = src_obj
        
        force_convert |= (src_obj.type != 'MESH')
        
        if force_convert:
            bpy.ops.object.convert(target='MESH', keep_original=True)
        else:
            # We use this rather than direct copy()
            # to piggyback on adding-to-scene logic
            bpy.ops.object.duplicate(linked=(mesh is not None))
        
        dst_obj = bpy.context.active_object
        
        if name is not None:
            dst_obj.name = name
        
        if mesh is not None:
            dst_obj.data = mesh
        
        if (not config.use_base_mesh) or force_convert:
            dst_obj.modifiers.clear()
        
        return dst_obj

    @classmethod
    def get_corner_normals(cls, mesh, config):
        if 'NORMAL' not in config.sharp_conditions:
            return None
        
        # Deprecated in Blender 4.1, but necessary in earlier versions:
        if hasattr(mesh, "calc_normals_split"):
            mesh.calc_normals_split()
        
        return [cn.vector.copy() for cn in mesh.corner_normals]

    @classmethod
    def get_mesh_data(cls, obj, depsgraph, config):
        bm = bmesh.new(use_operators=True)
        
        if (not config.use_base_mesh) or (obj.type != 'MESH'):
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh(preserve_all_data_layers=False)
            corner_normals = cls.get_corner_normals(mesh, config)
            bm.from_mesh(mesh)
            obj_eval.to_mesh_clear()
        else:
            mesh = obj.data
            corner_normals = cls.get_corner_normals(mesh, config)
            bm.from_mesh(mesh)
        
        return bm, corner_normals

    @classmethod
    def export_geometry(cls, path, bm):
        lines = []
        lines_append = lines.append
        
        for i, v in enumerate(bm.verts):
            v.index = i + 1 # OBJ indices start at 1
            x, y, z = v.co
            lines_append(f'v {x} {y} {z}\n')
        
        for f in bm.faces:
            v0, v1, v2 = f.verts
            lines_append(f'f {v0.index} {v1.index} {v2.index}\n')
            
            for i, l in enumerate(f.loops):
                l.index = i
        
        with open(path, 'w') as file:
            file.writelines(lines)
        
        ##################################################
        
        sharp_path = os.path.splitext(path)[0] + '.sharp'
        
        bm.faces.index_update()
        
        sharp_features = []
        for e in bm.edges:
            # Shouldn't happen, but can occur for now (not sure why)
            if e.is_wire:
                continue
            
            if e.smooth:
                continue
            
            loop = e.link_loops[0]
            angle = e.calc_face_angle_signed(math.pi)
            convexity = (0 if angle < 0 else 1)
            face_index = loop.face.index
            edge_index = loop.index
            sharp_features.append((convexity, face_index, edge_index))
        
        lines = [f'{len(sharp_features)}\n']
        for convexity, face_index, edge_index in sharp_features:
            lines.append(f'{convexity}, {face_index}, {edge_index}\n')
        
        with open(sharp_path, 'w') as file:
            file.writelines(lines)
        
        ##################################################
        
        # So far, it seems that we don't need to compute
        # a custom Rotational Symmetry field (quad flows
        # can be directed well enough with sharp edges)
        
        rosy_layer = bm.faces.layers.float_vector.get(RoSy.layer_name)
        
        if rosy_layer:
            rosy_path = os.path.splitext(path)[0] + '.rosy'
            
            lines = [f'{len(bm.faces)}\n', '4\n']
            for f in bm.faces:
                x, y, z = f[rosy_layer]
                lines.append(f'{x} {y} {z}\n')
            
            with open(rosy_path, 'w') as file:
                file.writelines(lines)

    @classmethod
    def read_polygon_count(cls, path):
        if not os.path.isfile(path):
            return 0
        
        with open(path, 'r') as file:
            lines = file.readlines()
        
        count = 0
        
        for line in lines:
            if line.startswith('f '):
                count += 1
                continue
            if not line.startswith('#'):
                continue
            
            line = line[1:].strip().lower()
            if not line.startswith('faces:'):
                continue
            
            parts = line.split(':')
            if len(parts) < 2:
                continue
            
            try:
                return int(parts[1].strip())
            except Exception:
                pass
        
        return count

    @classmethod
    def import_geometry(cls, path, bm):
        with open(path, 'r') as file:
            lines = file.readlines()
        
        verts = []
        verts_append = verts.append
        
        faces = []
        faces_append = faces.append
        
        verts_new = bm.verts.new
        faces_new = bm.faces.new
        
        for line in lines:
            if line.startswith('#'):
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            key = parts.pop(0)
            if key == 'v':
                x, y, z = parts
                verts_append(verts_new((float(x), float(y), float(z))))
            elif key == 'f':
                try:
                    faces_append(faces_new((verts[int(i)-1] for i in parts)))
                except Exception:
                    pass # same face already added
        
        return verts, faces

    @classmethod
    def postprocess_geometry(cls, bm, new_faces):
        pass

    def __init__(self, config, objects):
        self.stop = False
        self.status = ''
        self.error = ''
        self.export_path = ''
        self.import_path = ''
        self.current_dir = os.getcwd()
        self.config = config
        
        self.output_lines = []
        
        self.src_objs = [obj for obj in objects if obj.type in self.object_types]
        self.dst_objs = [None] * len(self.src_objs)
        self.obj_index = -1
        
        self.depsgraph = None

    def restore_current_dir(self):
        os.chdir(self.current_dir)

    def export_mesh(self):
        self.status = 'Exporting'
        
        if os.path.exists(self.tmp_path):
            # Make sure we don't have any leftovers from the previous run
            for filename in os.listdir(self.tmp_path):
                os.remove(os.path.join(self.tmp_path, filename))
        else:
            os.makedirs(os.path.join(self.tmp_path, ''))
        
        self.export_path = os.path.join(self.tmp_path, self.mesh_names[0])
        
        obj = self.src_objs[self.obj_index]
        
        bm, corner_normals = self.get_mesh_data(obj, self.depsgraph, self.config)
        
        bm_export = self.preprocess_geometry(bm, corner_normals, self.config)
        
        self.export_geometry(self.export_path, bm_export)
        
        bm_export.free() # either same as bm, or a copy of edit-mode bm
        
        return True

    def run_prep(self, detail):
        self.status = 'Prep stage'
        
        config_path = os.path.join(self.tmp_path, 'prep_config.txt')
        
        with open(config_path, 'w') as file:
            file.write(self.make_configs(self.config, detail=detail)['prep'])
        
        mesh_path = self.last_mesh_path()[1]
        args = [self.prep_path, mesh_path, '2', config_path]
        
        rosy_path = os.path.splitext(mesh_path)[0] + '.rosy'
        if os.path.isfile(rosy_path):
            args.append(rosy_path)
        
        sharp_path = os.path.splitext(mesh_path)[0] + '.sharp'
        if os.path.isfile(sharp_path):
            args.append(sharp_path)
        
        try:
            return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            self.error = f'Failed to execute prep stage of quadwild\n{exc}'

    def run_main(self, detail):
        self.status = 'Main stage'
        
        config_path = os.path.join(self.tmp_path, 'main_config.txt')
        
        with open(config_path, 'w') as file:
            file.write(self.make_configs(self.config, detail=detail)['main'])
        
        mesh_path = self.last_mesh_path()[1]
        args = [self.main_path, mesh_path, '1', config_path]
        
        try:
            return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except Exception as exc:
            self.error = f'Failed to execute main stage of quadwild\n{exc}'

    def import_mesh(self):
        self.status = 'Importing'
        
        self.import_path = self.last_mesh_path(1)[1]
        
        if not self.import_path:
            self.error = 'No mesh processing was performed'
            return False
        
        src_obj = self.src_objs[self.obj_index]
        name = f'{src_obj.name} (remeshed)'
        
        mesh = bpy.data.meshes.new(name)
        bm = bmesh.new(use_operators=True)
        
        verts, faces = self.import_geometry(self.import_path, bm)
        self.postprocess_geometry(bm, faces)
        
        bm.to_mesh(mesh)
        bm.free()
        
        dst_obj = self.copy_as_mesh(src_obj, self.config, name=name, mesh=mesh)
        
        self.dst_objs[self.obj_index] = dst_obj
        
        return True

    def optimize_polycount(self, timeout, detail, iterations=2):
        sqrt_target = math.sqrt(self.config.polycount)
        
        xx_sum = 0.0
        xy_sum = 0.0
        
        for iteration in range(iterations):
            actual_count = self.read_polygon_count(self.last_mesh_path(1)[1])
            
            if actual_count <= 0:
                return
            
            for mesh_name in self.mesh_names[-2:]:
                os.remove(os.path.join(self.tmp_path, mesh_name))
            
            # Number of polygons scales as the square of detail
            sqrt_count = math.sqrt(actual_count)
            xx_sum += actual_count
            xy_sum += sqrt_count * detail
            k = xy_sum / xx_sum
            
            detail = sqrt_target * (xy_sum / xx_sum)
            
            yield from self.await_finish(self.run_main(detail), timeout)
            
            if self.stop:
                return

    def hide_original_object(self):
        self.src_objs[self.obj_index].hide_set(True)

    def await_finish(self, process, timeout):
        if not process:
            return
        
        if timeout is not None:
            process.wait(timeout if timeout > 0 else None)
            return
        
        def read_output():
            while True:
                text = process.stdout.readline()
                if not text:
                    break
                
                if self.output_lines and (self.output_lines[-1][-1] != '\n'):
                    self.output_lines[-1] += text
                else:
                    self.output_lines.append(text)
                
                break # distribute line reading over time
        
        while not self.stop:
            if process.poll() is None:
                read_output()
                yield
            else:
                return
        
        process.terminate()

    def process(self, timeout=None):
        if self.stop:
            return
        
        # Flow config paths are relative to the bin directory
        os.chdir(self.bin_path)
        
        self.depsgraph = bpy.context.evaluated_depsgraph_get()
        
        for obj_index in range(len(self.src_objs)):
            self.obj_index = obj_index
            yield from self.process_object(timeout)
            
            if self.stop:
                return
        
        for obj in self.dst_objs:
            if obj:
                obj.select_set(True)

    def process_object(self, timeout):
        self.output_lines.append('\n')
        self.output_lines.append('Preprocessing...\n')
        self.output_lines.append('\n')
        yield
        
        if not self.export_mesh():
            return
        
        if self.config.detail_mode == 'DETAIL':
            detail = self.config.detail
        elif self.config.detail_mode == 'POLYCOUNT':
            detail = 1.0
        
        mesh_index = 0
        
        # Note: quadwild programs may return non-zero exit status
        # even when output was created, so we should't rely on it
        
        # PREP STAGE
        ################################################################
        yield from self.await_finish(self.run_prep(detail), timeout)
        
        if self.stop:
            return
        
        new_mesh_index = self.last_mesh_path()[0]
        
        if new_mesh_index == mesh_index:
            self.error = 'Prep stage failed'
            return
        
        mesh_index = new_mesh_index
        ################################################################
        
        # MAIN STAGE
        ################################################################
        yield from self.await_finish(self.run_main(detail), timeout)
        
        if self.stop:
            return
        
        new_mesh_index = self.last_mesh_path()[0]
        
        if new_mesh_index == mesh_index:
            self.error = 'Main stage failed'
            return
        
        if self.config.detail_mode == 'POLYCOUNT':
            yield from self.optimize_polycount(timeout, detail)
        
        mesh_index = new_mesh_index
        ################################################################
        
        if not self.import_mesh():
            return
        
        self.hide_original_object()


class RETOPO_OT_quadwild_preprocess(bpy.types.Operator):
    bl_idname = "retopo.quadwild_preprocess"
    bl_label = "Preprocess"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Apply mesh preprocessing"

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT'}
        # return context.mode in {'OBJECT', 'EDIT_MESH'} # mesh mode not implemented yet

    def execute(self, context):
        config = context.scene.retopo.ops.quadwild.config
        
        force_convert = not config.use_base_mesh
        
        src_objs = list(context.selected_objects)
        dst_objs = []
        for src_obj in src_objs:
            dst_obj = QuadWild.copy_as_mesh(src_obj, config, force_convert=force_convert)
            dst_obj.modifiers.clear() # remove modifiers in any case
            src_obj.hide_set(True)
            dst_objs.append(dst_obj)
        
        depsgraph = context.evaluated_depsgraph_get()
        
        use_preprocess = config.use_preprocess
        config.use_preprocess = True
        
        for dst_obj in dst_objs:
            bm, corner_normals = QuadWild.get_mesh_data(dst_obj, depsgraph, config)
            bm_processed = QuadWild.preprocess_geometry(bm, corner_normals, config)
            bm_processed.to_mesh(dst_obj.data)
            bm_processed.free()
        
        config.use_preprocess = use_preprocess
        
        return {'FINISHED'}


class RETOPO_OT_quadwild(bpy.types.Operator):
    bl_idname = "retopo.quadwild"
    bl_label = "QuadWild retopology"
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Perform retopology using the QuadWild algorithm"

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT'}
        # return context.mode in {'OBJECT', 'EDIT_MESH'} # mesh mode not implemented yet

    @classmethod
    def get_selection(cls, context):
        if context.mode == 'OBJECT':
            return [obj for obj in context.selected_objects if obj.type in QuadWild.object_types]
        elif context.mode == 'EDIT_MESH':
            obj = context.active_object
            if (not obj) or (obj.type != 'MESH'): return False
            bm = bmesh.from_edit_mesh(obj.data)
            return [f for f in bm.faces if f.select]

    def initialize(self, context, is_modal):
        selection = self.get_selection(context)
        
        if not selection:
            self.report({'WARNING'}, 'No objects or polygons selected')
            return False
        
        timeout = (None if is_modal else -1)
        config = context.scene.retopo.ops.quadwild.config
        objects = context.selected_objects
        
        self.quadwild_processor = QuadWild(config, objects)
        self.quadwild_iterator = self.quadwild_processor.process(timeout)
        
        return True

    def invoke(self, context, event):
        if not self.initialize(context, is_modal=True):
            return {'CANCELLED'}
        
        self.draw_handler = bpy.types.SpaceView3D.draw_handler_add(self.draw_2d, (), 'WINDOW', 'POST_PIXEL')
        context.area.tag_redraw()
        
        wm = context.window_manager
        self.timer = wm.event_timer_add(0.025, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self.quadwild_processor.stop = True
        elif event.type != 'TIMER':
            return {'RUNNING_MODAL'}
        
        is_finished = next(self.quadwild_iterator, True)
        
        status = self.quadwild_processor.status
        context.workspace.status_text_set(f'QuadWild: {status}... (ESC to cancel)')
        
        context.area.tag_redraw()
        
        if is_finished:
            bpy.types.SpaceView3D.draw_handler_remove(self.draw_handler, 'WINDOW')
            
            wm = context.window_manager
            wm.event_timer_remove(self.timer)
            
            context.workspace.status_text_set(None)
            
            self.quadwild_processor.restore_current_dir()
            
            if self.quadwild_processor.error:
                self.report({'ERROR'}, self.quadwild_processor.error)
            
            is_aborted = bool(self.quadwild_processor.error) or self.quadwild_processor.stop
            return {'CANCELLED' if is_aborted else 'FINISHED'}
        
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.initialize(context, is_modal=False):
            return {'CANCELLED'}
        
        for _ in self.quadwild_iterator:
            pass
        
        self.quadwild_processor.restore_current_dir()
        
        if self.quadwild_processor.error:
            self.report({'ERROR'}, self.quadwild_processor.error)
            return {'CANCELLED'}
        
        return {'FINISHED'}

    @staticmethod
    def calc_region_rect(area, r, overlap=True, convert=None):
        # Note: there may be more than one region of the same type (e.g. in quadview)
        if (not overlap) and (r.type == 'WINDOW'):
            x0, y0, x1, y1 = r.x, r.y, r.x+r.width, r.y+r.height
            
            for r in area.regions:
                if (r.width <= 0) or (r.height <= 0): continue
                
                # A HUD-specific hack. HUD in 3d view in some cases does not
                # become 1x1 when it's "hidden", but we may still attempt to
                # detect it by its (x,y) being zero
                if (r.alignment == 'FLOAT') and (r.x == 0) and (r.y == 0): continue
                
                alignment = r.alignment
                if convert: alignment = convert.get(alignment, alignment)
                
                if alignment == 'TOP':
                    y1 = min(y1, r.y)
                elif alignment == 'BOTTOM':
                    y0 = max(y0, r.y + r.height)
                elif alignment == 'LEFT':
                    x0 = max(x0, r.x + r.width)
                elif alignment == 'RIGHT':
                    x1 = min(x1, r.x)
            
            return x0, y0, x1-x0, y1-y0
        else:
            return r.x, r.y, r.width, r.height

    def draw_2d(self):
        context = bpy.context
        area = context.area
        region = context.region
        
        # prefs = context.preferences.system
        # ui_scale = prefs.dpi / BlUI.DPI
        
        x, y, width, height = self.calc_region_rect(area, region, overlap=False, convert={'FLOAT':'LEFT'})
        x -= region.x
        y -= region.y
        
        font_id = 0
        
        font_size = 24.0
        
        blf.enable(font_id, blf.WORD_WRAP)
        blf.word_wrap(font_id, width)
        blf.size(font_id, font_size)
        
        text = "".join(self.quadwild_processor.output_lines[-256:])
        
        size_x, size_y = blf.dimensions(font_id, text)
        y += size_y
        
        xmin, ymin, xmax, ymax = 0, 0, region.width, region.height
        
        gpu.state.blend_set('ALPHA')
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        coords = [
            (xmin, ymin, 0), (xmin, ymax, 0), (xmax, ymax, 0),
            (xmin, ymin, 0), (xmax, ymax, 0), (xmax, ymin, 0),
        ]
        batch = batch_for_shader(shader, 'TRIS', {"pos": coords})
        shader.uniform_float("color", (0, 0, 0, 0.75))
        batch.draw(shader)
        
        blf.position(font_id, x, y, 0)
        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.draw(font_id, text)


class RETOPO_OT_quadwild_install(bpy.types.Operator):
    bl_idname = "retopo.quadwild_install"
    bl_label = "Install QuadWild"
    bl_options = {'INTERNAL'}
    bl_description = "Download and unpack the QuadWild library"

    def execute(self, context):
        QuadWild.install()
        
        if QuadWild.installation_error:
            self.report({'ERROR'}, QuadWild.installation_error)
            return {'CANCELLED'}
        
        return {'FINISHED'}


def register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if not kc: return
    
    km = kc.keymaps.new(name='3D View', space_type='VIEW_3D')
    
    kmi = km.keymap_items.new("retopo.rosy_combing", 'LEFTMOUSE', 'PRESS')
    kmi = km.keymap_items.new("retopo.rosy_mouse_detect", 'MOUSEMOVE', 'ANY')


def unregister_keymaps():
    kc = bpy.context.window_manager.keyconfigs.addon
    if not kc: return
    
    op_idnames = ["retopo.rosy_combing", "retopo.rosy_mouse_detect"]
    menu_idnames = []
    panel_idnames = []
    
    for km in kc.keymaps:
        for kmi in tuple(km.keymap_items):
            if kmi.idname in op_idnames:
                km.keymap_items.remove(kmi)
            elif (kmi.idname in ("wm.call_menu", "wm.call_menu_pie")) and (kmi.properties.name in menu_idnames):
                km.keymap_items.remove(kmi)
            elif (kmi.idname == "wm.call_panel") and (kmi.properties.name in panel_idnames):
                km.keymap_items.remove(kmi)


draw_handlers = []


def add_draw_handler(callback, args, region_type, event):
    handler = bpy.types.SpaceView3D.draw_handler_add(callback, args, region_type, event)
    draw_handlers.append((handler, region_type))


def register_draw_handlers():
    add_draw_handler(RoSy.draw_2d, (), 'WINDOW', 'POST_PIXEL')
    add_draw_handler(RoSy.draw_3d, (), 'WINDOW', 'POST_VIEW')


def unregister_draw_handlers():
    for handler, region_type in draw_handlers:
        bpy.types.SpaceView3D.draw_handler_remove(handler, region_type)
    
    draw_handlers.clear()


event_handlers = []


def add_event_handler(event_name, callback, persistent=True):
    if persistent:
        @bpy.app.handlers.persistent
        def persistent_callback(scene):
            callback(scene)
        actual_callback = persistent_callback
    else:
        actual_callback = callback
    
    getattr(bpy.app.handlers, event_name).append(actual_callback)
    event_handlers.append((event_name, actual_callback))


def register_event_handlers():
    add_event_handler('undo_post', RoSy.on_data_change)
    add_event_handler('redo_post', RoSy.on_data_change)
    add_event_handler('load_post', RoSy.on_data_change)
    add_event_handler('frame_change_post', RoSy.on_data_change)
    add_event_handler('depsgraph_update_post', RoSy.on_depsgraph_update)


def unregister_event_handlers():
    for event_name, callback in event_handlers:
        getattr(bpy.app.handlers, event_name).remove(callback)
    event_handlers.clear()


def register():
    QuadWild.initialize()
    register_keymaps()
    register_event_handlers()
    register_draw_handlers()
    RoSy.on_startup()


def unregister():
    unregister_draw_handlers()
    unregister_event_handlers()
    unregister_keymaps()


types_classes = (
    RoSyOptions,
    QuadWildConfig,
    types.scene,
)

classes = (
    RETOPO_OT_rosy_mouse_detect,
    RETOPO_OT_rosy_reset,
    RETOPO_OT_rosy_relax,
    RETOPO_OT_rosy_combing,
    RETOPO_OT_quadwild_preprocess,
    RETOPO_OT_quadwild,
    RETOPO_OT_quadwild_install,
)