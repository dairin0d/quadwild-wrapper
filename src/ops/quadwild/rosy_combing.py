import math
import time

import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.geometry import tessellate_polygon, closest_point_on_tri
from bpy_extras import view3d_utils

import gpu
from gpu_extras.batch import batch_for_shader

from .utils import PrimitiveLock, is_valid
from .geom_curvature import calc_rosy, rosy_align, rosy_smooth


"""
TODO:
* offset from the surface a bit?
* display the mode as toggles instead of dropdown?
* remeshing brush?
* shortcuts for changing the size / angle / strength of the brush?
* a button to preprocess the mesh (e.g. to mark the sharp edges)
"""


class RoSyOptions(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name='RoSy combing',
        description='Enable flow painting/combing',
        default=False,
        update=(lambda self, context: RoSy.synchronize_state(force=True)))

    mode: bpy.props.EnumProperty(
        name='Mode',
        description='Flow painting/combing mode',
        items=[
            ('DRAG', 'Drag', 'Align flow direction to the direction of brush stroke'),
            ('ALIGN', 'Align', 'Align flow direction to match a specific orientation'),
            ('ATTRACT', 'Attract', 'Align flow direction toward a specific point'),
            ('SMOOTH', 'Smooth', 'Align flow directions between adjacent polygons'),
        ],
        default='DRAG')

    brush_size: bpy.props.FloatProperty(
        name='Brush size',
        description='Brush size (in pixels)',
        min=1,
        default=50)

    brush_angle: bpy.props.FloatProperty(
        name='Brush angle',
        description='Brush angle (to which the flow directions will be aligned in Align mode)',
        subtype='ANGLE',
        default=0)

    brush_strength: bpy.props.FloatProperty(
        name='Brush strength',
        description='Brush strength',
        default=1)


class RETOPO_OT_rosy_mouse_detect(bpy.types.Operator):
    bl_idname = "retopo.rosy_mouse_detect"
    bl_label = "Mouse detect"
    bl_options = {'INTERNAL'}
    bl_description = ""

    def invoke(self, context, event):
        RoSy.update_mouse(context, event)
        return {'PASS_THROUGH'}


class RosyOperator(bpy.types.Operator):
    @classmethod
    def poll(cls, context):
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        
        if not rosy_options.enabled:
            return False
        
        if not context.object:
            return False
        
        return context.mode in {'OBJECT'}
        # return context.mode in {'OBJECT', 'EDIT_MESH'} # mesh mode not implemented yet


class RETOPO_OT_rosy_reset(RosyOperator):
    bl_idname = "retopo.rosy_reset"
    bl_label = "RoSy Reset"
    bl_options = {'INTERNAL'}
    bl_description = "Reset (recalculate) the flow directions"

    def execute(self, context):
        RoSy.reset()
        context.area.tag_redraw()
        return {'PASS_THROUGH'}


class RETOPO_OT_rosy_relax(RosyOperator):
    bl_idname = "retopo.rosy_relax"
    bl_label = "RoSy Relax"
    bl_options = {'INTERNAL'}
    bl_description = "Relax (smoothen) the flow directions"

    def execute(self, context):
        RoSy.smooth()
        context.area.tag_redraw()
        return {'PASS_THROUGH'}


class RETOPO_OT_rosy_combing(RosyOperator):
    bl_idname = "retopo.rosy_combing"
    bl_label = "RoSy combing"
    bl_options = {'INTERNAL'}
    bl_description = "Customize the flow directions (Rotational Symmetry field)"

    def invoke(self, context, event):
        RoSy.synchronize_state()
        
        self.last_pos = None
        self.drag_direction = Vector()
        
        wm = context.window_manager
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'ESC':
            self.exit(context, cancel=True)
            return {'CANCELLED'}
        
        if (event.type == 'LEFTMOUSE') and (event.value == 'RELEASE'):
            self.exit(context, cancel=False)
            return {'FINISHED'}
        
        if event.type != 'MOUSEMOVE':
            return {'RUNNING_MODAL'}
        
        RoSy.update_mouse(context, event)
        
        region = context.region
        rv3d = context.region_data
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        
        coord = Vector((event.mouse_region_x, event.mouse_region_y))
        success, location, normal, face_index = RoSy.ray_cast(region, rv3d, coord)
        
        if not success:
            self.last_pos = None
            self.drag_direction = Vector()
            return {'RUNNING_MODAL'}
        
        matrix = RoSy.obj.matrix_world
        center = matrix @ location
        radius = RoSy.calc_influence_radius(region, rv3d, coord, center, rosy_options.brush_size)
        
        drag_direction = None
        if self.last_pos is not None:
            drag_direction = center - self.last_pos
        
        self.last_pos = center
        
        if drag_direction is not None:
            brush_strength = rosy_options.brush_strength
            brush_strength *= min(100 * drag_direction.magnitude / radius, 1.0)
            
            face, verts, tris, tris_count = RoSy.data[face_index]
            faces = RoSy.find_affected_faces(face, center, radius)
            
            if rosy_options.mode == 'DRAG':
                drag_lerp = 0.1
                self.drag_direction = self.drag_direction.lerp(drag_direction, drag_lerp)
                drag_direction = self.drag_direction
                for face, influence in faces.items():
                    RoSy.align(face, drag_direction, influence * brush_strength)
            elif rosy_options.mode == 'ALIGN':
                angle = rosy_options.brush_angle
                vector_ui = Vector((math.sin(angle), math.cos(angle), 0))
                vector_3d = RoSy.from_screenspace(vector_ui)
                for face, influence in faces.items():
                    normal = face.normal
                    RoSy.align(face, vector_3d, influence * brush_strength)
            elif rosy_options.mode == 'ATTRACT':
                for face, influence in faces.items():
                    direction = location - face.calc_center_median_weighted()
                    RoSy.align(face, direction, influence * brush_strength)
            elif rosy_options.mode == 'SMOOTH':
                faces, factors = list(faces.keys()), [factor * brush_strength for factor in faces.values()]
                RoSy.smooth(faces, factors)
            
            region.tag_redraw()
        
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        if not cancel:
            RoSy.bm.to_mesh(RoSy.mesh)
            bpy.ops.ed.undo_push(message='RoSy flow combing')


class RoSy:
    layer_name = 'RoSy'

    mouse_info = {'x': 0, 'y': 0, 'time': -math.inf}

    recursion_lock = PrimitiveLock()

    obj = None
    mesh = None
    bm = None
    layer = None
    data = None
    rosy_pos = None
    rosy_color = None
    batches = None
    
    batch_size = 256
    dirty_batches = set()

    color_center = Vector((0.5, 0.5, 0.5))

    @classmethod
    def to_color(cls, v):
        return (v * 0.5) + cls.color_center

    @classmethod
    def from_screenspace(cls, v):
        context = bpy.context
        rv3d = context.region_data
        
        if not (cls.obj and isinstance(rv3d, bpy.types.RegionView3D)):
            return Vector()
        
        # rv3d.view_matrix is world->view
        v = rv3d.view_matrix.to_3x3().inverted_safe() @ v
        v = cls.obj.matrix_world.to_3x3().inverted_safe() @ v
        return v.normalized()

    @classmethod
    def is_valid(cls):
        if (cls.obj is not None) and (not is_valid(cls.obj)):
            return False
        if (cls.mesh is not None) and (not is_valid(cls.mesh)):
            return False
        return True

    @classmethod
    def synchronize_state(cls, force=False):
        context = bpy.context
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        
        if rosy_options.enabled:
            cls.refresh(force)
        else:
            cls.cleanup()

    @classmethod
    def ensure_initialized(cls):
        context = bpy.context
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        
        if not rosy_options.enabled:
            return False
        
        if not (cls.bm and cls.layer):
            cls.refresh()
        
        return True

    @classmethod
    def reset(cls):
        if not cls.ensure_initialized():
            return
        
        if cls.bm and cls.layer:
            calc_rosy(cls.bm.faces, cls.layer)
            cls.bm.to_mesh(cls.mesh)
            bpy.ops.ed.undo_push(message='Initialize RoSy data')
            cls.update_polygons()

    @classmethod
    def refresh(cls, force=False):
        context = bpy.context
        obj = context.object
        mesh = (obj.data if obj and (obj.type == 'MESH') else None)
        
        if force or (obj is not cls.obj) or (mesh is not cls.mesh) or not cls.is_valid():
            cls.cleanup()
        
        if not mesh:
            return
        
        if (cls.bm is None) or (cls.layer is None):
            bm = bmesh.new(use_operators=True)
            bm.from_mesh(mesh)
            layer = bm.faces.layers.float_vector.get(cls.layer_name)
            
            if not layer:
                layer = bm.faces.layers.float_vector.new(cls.layer_name)
                calc_rosy(bm.faces, layer)
                bm.to_mesh(mesh)
                bpy.ops.ed.undo_push(message='Initialize RoSy data')
            
            cls.bm = bm
            cls.layer = layer
        
        cls.obj = obj
        cls.mesh = mesh
        
        if cls.data is None:
            data, tris_count = cls.build_data()
            cls.data = data
            cls.rosy_pos = [None] * (tris_count * 4)
            cls.rosy_color = [None] * (tris_count * 4)
            cls.update_polygons()

    @classmethod
    def build_data(cls):
        data = [None] * len(cls.bm.faces)
        tris_count = 0
        
        for i, f in enumerate(cls.bm.faces):
            f.index = i
            verts = [v.co.copy() for v in f.verts]
            tris = tessellate_polygon([verts])
            data[i] = (f, verts, tris, tris_count)
            tris_count += len(tris)
        
        return data, tris_count

    @classmethod
    def smooth(cls, faces=None, factors=0.25):
        if faces is not None:
            rosy_smooth(faces, cls.layer, factors)
            
            for face in faces:
                cls.update_polygon(face.index)
            
            return
        
        if not cls.ensure_initialized():
            return
        
        rosy_smooth(cls.bm.faces, cls.layer)
        cls.bm.to_mesh(cls.mesh)
        bpy.ops.ed.undo_push(message='Smooth RoSy directions')
        cls.update_polygons()

    @classmethod
    def update_polygons(cls):
        for i in range(len(cls.data)):
            cls.update_polygon(i)

    @classmethod
    def incircle(cls, v0, v1, v2):
        a = (v2 - v1).magnitude
        b = (v0 - v2).magnitude
        c = (v1 - v0).magnitude
        perimeter = a + b + c
        s = perimeter*0.5
        radius = math.sqrt((s-a)*(s-b)*(s-c)/s)
        center = (a*v0 + b*v1 + c*v2) / perimeter
        return center, radius

    @classmethod
    def midcircle(cls, v0, v1, v2):
        center = (v0 + v1 + v2) / 3.0
        r0 = (v1 - v0).normalized().cross(center - v0).magnitude
        r1 = (v2 - v1).normalized().cross(center - v1).magnitude
        r2 = (v0 - v2).normalized().cross(center - v2).magnitude
        return center, min(r0, r1, r2)

    @classmethod
    def update_polygon(cls, polygon_index):
        face, verts, tris, tri_start = cls.data[polygon_index]
        
        rosy_dir = face[cls.layer].normalized()
        
        rosy_pos = cls.rosy_pos
        rosy_color = cls.rosy_color
        
        for tri_index in range(len(tris)):
            batch_index = (tri_start + tri_index) * 4
            i0, i1, i2 = tris[tri_index]
            v0, v1, v2 = verts[i0], verts[i1], verts[i2]
            
            normal = (v1 - v0).cross(v2 - v0)
            normal.normalize()
            
            center, scale = cls.midcircle(v0, v1, v2)
            
            norm_x = rosy_dir.cross(normal).normalized()
            norm_y = norm_x.cross(normal)
            axis_x = norm_x * scale
            axis_y = norm_y * scale
            
            rosy_pos[batch_index+0] = center-axis_x
            rosy_pos[batch_index+1] = center+axis_x
            rosy_pos[batch_index+2] = center-axis_y
            rosy_pos[batch_index+3] = center+axis_y
            
            rosy_color[batch_index+0] = cls.to_color(-norm_x)
            rosy_color[batch_index+1] = cls.to_color(norm_x)
            rosy_color[batch_index+2] = cls.to_color(-norm_y)
            rosy_color[batch_index+3] = cls.to_color(norm_y)
        
        batch_size = cls.batch_size * 4 # 4 verts per element
        batch_index_min = tri_start * 4 // batch_size
        batch_index_max = (tri_start + len(tris)) * 4 // batch_size
        cls.dirty_batches.update(range(batch_index_min, batch_index_max+1))

    @classmethod
    def build_batches(cls):
        if not cls.batches:
            cls.batches = []
        
        batch_size = cls.batch_size * 4 # 4 verts per element
        batch_count = (len(cls.rosy_pos) + batch_size-1) // batch_size
        
        if len(cls.batches) != batch_count:
            cls.batches = [None] * batch_count
            indices = range(batch_count)
        else:
            if not cls.dirty_batches:
                return
            
            indices = cls.dirty_batches
        
        shader = gpu.shader.from_builtin('SMOOTH_COLOR')
        
        for batch_index in indices:
            if (batch_index < 0) or (batch_index >= batch_count):
                continue
            
            index_min = batch_index * batch_size
            index_max = index_min + batch_size
            pos = cls.rosy_pos[index_min:index_max]
            color = cls.rosy_color[index_min:index_max]
            
            batch = batch_for_shader(shader, 'LINES', {'pos': pos, 'color': color})
            batch.program_set(shader)
            
            cls.batches[batch_index] = batch
        
        cls.dirty_batches.clear()

    @classmethod
    def cleanup(cls):
        if cls.bm:
            cls.bm.free()
        
        cls.obj = None
        cls.mesh = None
        cls.bm = None
        cls.layer = None
        cls.data = None
        cls.rosy_pos = None
        cls.rosy_color = None
        cls.batches = None
        
        cls.dirty_batches = set()

    @classmethod
    def ray_cast(cls, region, rv3d, coord):
        ray_dir = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_min = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_max = ray_min + ray_dir
        
        matrix = cls.obj.matrix_world
        matrix_inv = matrix.inverted_safe()
        ray_min = matrix_inv @ ray_min
        ray_max = matrix_inv @ ray_max
        ray_dir = ray_max - ray_min
        
        return cls.obj.ray_cast(ray_min, ray_dir)

    @classmethod
    def calc_influence_radius(cls, region, rv3d, coord, center, brush_size):
        radius = 0.0
        for axis_index in (0, 1):
            for sign in (-1, 1):
                new_coord = Vector(coord)
                new_coord[axis_index] += sign * brush_size
                new_pos = view3d_utils.region_2d_to_location_3d(region, rv3d, new_coord, center)
                radius += (new_pos - center).magnitude
        radius /= 4
        return radius

    @classmethod
    def find_affected_faces(cls, face, center, radius):
        result = {}
        processed = {face}
        queue = [face]
        
        matrix = cls.obj.matrix_world
        
        abs_verts = [None] * 32
        
        def calc_influence(f):
            _, verts, tris, _ = cls.data[f.index]
            
            count_delta = len(verts) - len(abs_verts)
            if count_delta > 0:
                abs_verts.extend(range(count_delta))
            
            for i, v in enumerate(verts):
                abs_verts[i] = matrix @ verts[i]
            
            is_overlapping = False
            influence = 0.0
            for i0, i1, i2 in tris:
                v0 = abs_verts[i0]
                v1 = abs_verts[i1]
                v2 = abs_verts[i2]
                
                p = (v0 + v1 + v2) / 3.0
                rel_dist = (p - center).magnitude / radius
                if rel_dist < 1.0:
                    influence = max(influence, (1.0 - rel_dist))
                
                p = closest_point_on_tri(center, v0, v1, v2)
                is_overlapping |= ((p - center).magnitude < radius)
            
            return is_overlapping, influence
        
        while queue:
            face = queue.pop()
            
            is_overlapping, influence = calc_influence(face)
            
            if not is_overlapping:
                continue
            
            if influence > 0:
                result[face] = influence
            
            for e in face.edges:
                for f in e.link_faces:
                    if f not in processed:
                        queue.append(f)
                        processed.add(f)
        
        return result

    @classmethod
    def align(cls, face, direction, influence):
        old_dir = face[cls.layer].normalized()
        new_dir = direction.cross(face.normal).normalized()
        
        if new_dir.length_squared == 0:
            return
        
        old_dir = rosy_align(old_dir, new_dir)
        face[cls.layer] = old_dir.lerp(new_dir, influence).normalized()
        
        cls.update_polygon(face.index)

    @classmethod
    def update_mouse(cls, context, event):
        mouse_info = cls.mouse_info
        mouse_info['x'] = event.mouse_x
        mouse_info['y'] = event.mouse_y
        mouse_info['time'] = time.perf_counter()
        context.region.tag_redraw()

    @classmethod
    def find_attribute(cls, mesh):
        for attribute in mesh.attributes:
            if attribute.name == cls.layer_name:
                if attribute.data_type == 'FLOAT_VECTOR':
                    if attribute.domain == 'FACE':
                        return attribute

    @classmethod
    def draw_2d(cls):
        context = bpy.context
        region = context.region
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        
        if not rosy_options.enabled:
            return
        
        mouse_info = cls.mouse_info
        
        # I don't know of any way to detect whether a modal operator is currently running
        # So here we hide the brush display simply by the time since the last update
        visible_duration = 0.1
        
        if (time.perf_counter() - mouse_info['time']) > visible_duration:
            return
        
        x = mouse_info['x'] - region.x
        y = mouse_info['y'] - region.y
        dx = rosy_options.brush_size
        dy = rosy_options.brush_size
        
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        
        coords = []
        subdivs = 33
        for i in range(subdivs):
            angle = 2 * math.pi * (i / (subdivs-1))
            coords.append((math.sin(angle)*dx + x, math.cos(angle)*dy + y, 0))
        
        batch = batch_for_shader(shader, 'LINES', {"pos": coords[:-1]})
        shader.uniform_float("color", (0, 0, 0, 1))
        batch.draw(shader)
        
        batch = batch_for_shader(shader, 'LINES', {"pos": coords[1:]})
        shader.uniform_float("color", (1, 1, 1, 1))
        batch.draw(shader)
        
        if rosy_options.mode == 'ALIGN':
            shader = gpu.shader.from_builtin('SMOOTH_COLOR')
            
            coords = []
            colors = []
            for i in (0, 1, 0.5, 1.5):
                angle = i * math.pi + rosy_options.brush_angle
                vector_ui = Vector((math.sin(angle)*dx, math.cos(angle)*dy, 0))
                vector_3d = cls.from_screenspace(vector_ui)
                coords.append(vector_ui + Vector((x, y, 0)))
                colors.append(cls.to_color(vector_3d))
            
            gpu.state.line_width_set(2)
            
            batch = batch_for_shader(shader, 'LINES', {"pos": coords[:-2], 'color': colors[:-2]})
            batch.draw(shader)
            
            batch = batch_for_shader(shader, 'LINES', {"pos": coords[2:], 'color': colors[2:]})
            batch.draw(shader)
            
            gpu.state.line_width_set(1)

    @classmethod
    def draw_3d(cls):
        context = bpy.context
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        
        if not rosy_options.enabled:
            return
        
        obj = context.object
        
        if obj is not cls.obj:
            return
        
        cls.build_batches()
        
        if not cls.batches:
            return
        
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.line_width_set(3)
        
        with gpu.matrix.push_pop():
            gpu.matrix.multiply_matrix(obj.matrix_world)
            for batch in cls.batches:
                batch.draw()

    @classmethod
    def on_startup(cls):
        def on_timer():
            cls.synchronize_state(force=True)
        bpy.app.timers.register(on_timer)

    @classmethod
    def on_data_change(cls, scene=None):
        with cls.recursion_lock:
            cls.synchronize_state(force=True)

    @classmethod
    def on_depsgraph_update(cls, scene=None):
        with cls.recursion_lock:
            bpy_Object = bpy.types.Object
            bpy_Mesh = bpy.types.Mesh
            
            needs_refresh = not cls.is_valid()
            
            context = bpy.context
            depsgraph = context.evaluated_depsgraph_get()
            
            for update in depsgraph.updates:
                id_data = update.id
                if isinstance(id_data, bpy_Object):
                    needs_refresh |= (id_data is cls.obj)
                elif isinstance(id_data, bpy_Mesh):
                    needs_refresh |= (id_data is cls.mesh)
            
            if not needs_refresh:
                return
            
            cls.synchronize_state(force=True)
