import math
import itertools

import numpy as np

from mathutils import Vector


class PatchVertex:
    def _opposite(self, loop):
        # This check is necessary in some degenerate cases
        edge = loop.edge
        if not (edge.is_contiguous and edge.smooth):
            return None
        
        loop2 = loop.link_loop_radial_prev
        return (loop2 if loop2 != loop else None)

    def __init__(self, loop):
        loops = []
        
        loop_next, loop_prev = loop, self._opposite(loop.link_loop_prev)
        
        while loop_prev and (loop_prev != loop_next):
            edge = loop_prev.edge
            if not (edge.is_contiguous and edge.smooth):
                break
            loops.insert(0, loop_prev)
            loop_prev = self._opposite(loop_prev.link_loop_prev)
        
        if loop_prev == loop_next:
            loop_prev = self._opposite(loop_next).link_loop_next
        
        while loop_next and (loop_next != loop_prev):
            loops.append(loop_next)
            edge = loop_next.edge
            if not (edge.is_contiguous and edge.smooth):
                break
            loop_next = self._opposite(loop_next).link_loop_next
        
        # Based on (DOI: 10.1080/10867651.1999.10487501)
        # "Weights for Computing Vertex Normals from Facet Normals" (2000)
        normal = Vector()
        v0 = loop.vert.co
        for l in loops:
            v1, v2 = l.link_loop_next.vert.co, l.link_loop_prev.vert.co
            e1, e2 = (v1 - v0), (v2 - v0)
            e1_e2_mag_squared = e1.length_squared * e2.length_squared
            if e1_e2_mag_squared > 0:
                normal += e1.cross(e2) / e1_e2_mag_squared
        normal.normalize()
        
        tangentX = normal.orthogonal()
        tangentX.normalize()
        tangentY = normal.cross(tangentX)
        
        self.axes = (tangentX, tangentY, normal)
        self.pos = v0
        self.loops = loops
        self.c0 = 0.0
        self.cD = 0.0
        self.c1 = 0.0
        self.cw = 0.0
        self.edge_len = 0.0


def calc_curvatures(faces, layer_edge=None, layer_rosy=None, edge_ratio=0.2, edge_min=0, edge_max=math.inf):
    # Based on (DOI: 10.1109/TDPVT.2004.1335277)
    # "Estimating curvatures and their derivatives on triangle meshes" (2004)
    
    patch_verts = []
    loop_pv_map = {}
    
    for f in faces:
        for l in f.loops:
            pv = loop_pv_map.get(l)
            if pv:
                continue
            
            pv = PatchVertex(l)
            patch_verts.append(pv)
            for l2 in pv.loops:
                loop_pv_map[l2] = pv
    
    pi = math.pi
    pi2 = math.pi/2
    
    for f in faces:
        f_pvs = [loop_pv_map[l] for l in f.loops]
        if len(f_pvs) > 3:
            f_pvs = f_pvs[:3]
        
        pv0, pv1, pv2 = f_pvs
        e0 = pv2.pos - pv1.pos
        e1 = pv0.pos - pv2.pos
        e2 = pv1.pos - pv0.pos
        dn0 = pv2.axes[-1] - pv1.axes[-1]
        dn1 = pv0.axes[-1] - pv2.axes[-1]
        dn2 = pv1.axes[-1] - pv0.axes[-1]
        
        u = e2.normalized()
        nf = e1.cross(u)
        nf.normalize()
        v = u.cross(nf)
        
        # II @ (e*u, e*v) = (dn*u, dn*v)
        # II is symmetric [[a, b], [b, c]], so has only 3 unique parameters
        # a*(e*u) + b*(e*v) = dn*u = a*eu + b*ev + c*0
        # b*(e*u) + c*(e*v) = dn*v = a*0  + b*eu + c*ev
        e0u, e0v = e0.dot(u), e0.dot(v)
        e1u, e1v = e1.dot(u), e1.dot(v)
        e2u, e2v = e2.dot(u), e2.dot(v)
        data_x = [[e0u, e0v, 0], [0, e0u, e0v],
                 [e1u, e1v, 0], [0, e1u, e1v],
                 [e2u, e2v, 0], [0, e2u, e2v]]
        data_y = [dn0.dot(u), dn0.dot(v), dn1.dot(u), dn1.dot(v), dn2.dot(u), dn2.dot(v)]
        a0, aD, a1 = np.linalg.lstsq(data_x, data_y, None)[0]
        
        if layer_rosy is not None:
            eig_val, eig_vec = np.linalg.eig([[a0, aD], [aD, a1]])
            # If both curvatures have the same sign and magnitude, there is no preferred direction
            if math.isclose(eig_val[0], eig_val[1], rel_tol=0.001):
                f[layer_rosy] = Vector()
            else:
                eig_max_i = (1 if abs(eig_val[1]) > abs(eig_val[0]) else 0)
                eig_uv = eig_vec[:, eig_max_i] * eig_val[eig_max_i]
                f[layer_rosy] = u * eig_uv[0] + v * eig_uv[1]
        
        if layer_edge is None:
            continue
        
        edges2 = (e0.dot(e0), e1.dot(e1), e2.dot(e2))
        angles = (e2.angle(-e1, 0), e0.angle(-e2, 0), e1.angle(-e0, 0))
        cotans = tuple(1/max(math.tan(angle), 1e-16) for angle in angles)
        area = f.calc_area()
        is_obtuse = any((angle >= pi2) for angle in angles)
        
        for i in (0, 1, 2):
            pv = f_pvs[i]
            
            # Based on (DOI: 10.1007/978-3-662-05105-4_2)
            # "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds" (2002)
            if not is_obtuse:
                w = (edges2[i-1]*cotans[i-1] + edges2[i-2]*cotans[i-2]) / 8
            else:
                w = area / (2 if angles[i] >= pi2 else 4)
            
            r = pv.axes[-1].rotation_difference(nf)
            up = r @ pv.axes[0]
            vp = r @ pv.axes[1]
            up_uf, up_vf = up.dot(u), up.dot(v)
            vp_uf, vp_vf = vp.dot(u), vp.dot(v)
            b0 = up_uf*(a0*up_uf + aD*up_vf) + up_vf*(aD*up_uf + a1*up_vf)
            bD = up_uf*(a0*vp_uf + aD*vp_vf) + up_vf*(aD*vp_uf + a1*vp_vf)
            b1 = vp_uf*(a0*vp_uf + aD*vp_vf) + vp_vf*(aD*vp_uf + a1*vp_vf)
            
            pv.c0 += b0 * w
            pv.cD += bD * w
            pv.c1 += b1 * w
            pv.cw += w
    
    if layer_edge is None:
        return
    
    for pv in patch_verts:
        c0, cD, c1 = pv.c0/pv.cw, pv.cD/pv.cw, pv.c1/pv.cw
        eig_val, eig_vec = np.linalg.eig([[c0, cD], [cD, c1]])
        curv_max = max(abs(eig_val[0]), abs(eig_val[1]))
        radius = (1.0/curv_max if curv_max > 0 else math.inf)
        pv.edge_len = min(max(radius * edge_ratio, edge_min), edge_max)
    
    len_max = 4/3
    len_min = 4/5
    
    for f in faces:
        for l in f.loops:
            pv = loop_pv_map.get(l)
            l[layer_edge] = (pv.edge_len * len_min, pv.edge_len, pv.edge_len * len_max)


def rosy_align(a, b):
    a90 = a.cross(a.cross(b).normalized())
    dot0 = a.dot(b)
    dot90 = a90.dot(b)
    return (a * math.copysign(1, dot0) if abs(dot0) >= abs(dot90) else a90 * math.copysign(1, dot90))


def rosy_smooth(faces, layer_rosy, factors=0.25, force_boundary=False, keep_boundary=True):
    directions = []
    
    for f in faces:
        is_boundary = False
        
        direction = Vector()
        
        for l in f.loops:
            e = l.edge
            
            if e.smooth and e.is_manifold and not e.is_boundary:
                f2 = l.link_loop_radial_prev.face
                delta = f2[layer_rosy]
            else:
                is_boundary = True
                if force_boundary:
                    v0, v1 = e.verts
                    delta = v1.co - v0.co
                else:
                    delta = Vector()
            
            delta = delta.cross(f.normal)
            
            if delta.length_squared > 0:
                delta.normalize()
                delta *= e.calc_length()
                direction += rosy_align(delta, direction)
        
        if is_boundary and keep_boundary:
            direction = Vector()
        
        directions.append(direction)
    
    if not hasattr(factors, "__len__"):
        factors = itertools.repeat(factors, len(faces))
    
    for f, factor, dir_new in zip(faces, factors, directions):
        if factor == 0:
            continue
        
        if dir_new.length_squared == 0:
            continue
        
        dir_old = f[layer_rosy]
        dir_old = dir_old.cross(f.normal)
        dir_old.normalize()
        if dir_old.length_squared > 0:
            dir_old = rosy_align(dir_old, dir_new)
            dir_new = dir_old.slerp(dir_new, factor, dir_new)
        
        f[layer_rosy] = dir_new


def calc_rosy(faces, layer_rosy):
    # RoSy directions from curvature
    calc_curvatures(faces, layer_rosy=layer_rosy)
    
    undefined_faces = []
    
    # RoSy directions boundaries / sharp guides
    for f in faces:
        direction = Vector()
        for e in f.edges:
            if e.is_boundary or not (e.smooth and e.is_manifold):
                v0, v1 = e.verts
                delta = v1.co - v0.co
                direction += rosy_align(delta, direction)
        
        if direction.length_squared == 0:
            direction = f[layer_rosy]
        
        direction.normalize()
        f[layer_rosy] = direction
        
        if direction.length_squared == 0:
            undefined_faces.append(f)
    
    # Propagate to faces which don't have the directions defined
    while undefined_faces:
        candidates = []
        
        for i in range(len(undefined_faces)-1, -1, -1):
            f = undefined_faces[i]
            
            # By construction, all undefined faces have non-sharp non-boundary manifold edges
            direction = Vector()
            for l in f.loops:
                f2 = l.link_loop_radial_prev.face
                delta = f2[layer_rosy] * f2.calc_area()
                direction += rosy_align(delta, direction)
            
            if direction.length_squared > 0:
                candidates.append((f, direction))
                undefined_faces.pop(i)
        
        if not candidates:
            break
        
        for f, direction in candidates:
            # Make sure it lies in the face's plane
            direction = direction.cross(f.normal)
            direction.normalize()
            f[layer_rosy] = direction
    
    # In case any undefined faces left, use arbitrary directions
    axis_z = Vector((0, 0, 1))
    for f in undefined_faces:
        normal = f.normal
        if abs(normal.z) > 0.9999:
            direction = normal.orthogonal()
        else:
            direction = normal.cross(axis_z)
        direction.normalize()
        f[layer_rosy] = direction
