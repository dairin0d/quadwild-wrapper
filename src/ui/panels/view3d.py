import bpy


class RETOPO_PT_RetopologyPanel(bpy.types.Panel):
    bl_label = "QuadWild"
    bl_description = "QuadWild retopology tools"
    bl_idname = "RETOPO_PT_RetopologyPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QuadWild"

    def draw(self, context):
        layout = self.layout
        
        scene = context.scene
        rosy_options = scene.retopo.ops.quadwild.rosy
        quadwild_config = scene.retopo.ops.quadwild.config
        
        if not quadwild_config.is_installed:
            quadwild_config.draw_installation_ui(layout)
            return
        
        layout.operator("retopo.quadwild", text="Run auto-retopology")
        
        def draw_detail_radiobutton(layout, value, prop_name):
            is_active = (quadwild_config.detail_mode == value)
            icon = ('RADIOBUT_ON' if is_active else 'RADIOBUT_OFF')
            layout.prop_enum(quadwild_config, "detail_mode", value, text='', icon=icon)
            
            row = layout.row(align=True)
            row.enabled = is_active
            row.prop(quadwild_config, prop_name)
        
        grid = layout.grid_flow(row_major=True, columns=2, align=True)
        draw_detail_radiobutton(grid, 'DETAIL', "detail")
        draw_detail_radiobutton(grid, 'POLYCOUNT', "polycount")
        
        layout.prop(quadwild_config, "use_base_mesh")
        
        row = layout.row()
        subrow = row.row(align=True)
        icon = ('CHECKBOX_HLT' if quadwild_config.use_preprocess else 'CHECKBOX_DEHLT')
        subrow.prop(quadwild_config, "use_preprocess", text="", icon=icon, toggle=True)
        subrow.popover("RETOPO_PT_RetopologyPreprocessingOptionsPanel")
        subrow = row.row(align=True)
        subrow.alignment = 'RIGHT'
        subrow.operator("retopo.quadwild_preprocess", text="Apply")
        
        layout.popover("RETOPO_PT_RetopologyEdgeOptionsPanel")
        
        layout.popover("RETOPO_PT_RetopologyAdvancedOptionsPanel")
        
        self.draw_rosy(layout, rosy_options)
    
    def draw_rosy(self, layout, rosy_options):
        box = layout.box()
        col = box.column()
        row = col.row()
        row.prop(rosy_options, "enabled", text="Edit flow")
        subrow = row.row()
        subrow.enabled = rosy_options.enabled
        subrow.prop(rosy_options, "mode", text="")
        row = col.row()
        row.enabled = rosy_options.enabled
        row.operator("retopo.rosy_reset", text="Reset")
        row.operator("retopo.rosy_relax", text="Relax")
        subcol = col.column(align=True)
        subcol.enabled = rosy_options.enabled
        subcol.prop(rosy_options, "brush_size")
        subcol.prop(rosy_options, "brush_angle")
        subcol.prop(rosy_options, "brush_strength")


class RETOPO_PT_RetopologyPreprocessingOptionsPanel(bpy.types.Panel):
    bl_label = "Preprocessing"
    bl_description = "Preprocessing options"
    bl_idname = "RETOPO_PT_RetopologyPreprocessingOptionsPanel"
    bl_space_type = 'CONSOLE'
    bl_region_type = 'WINDOW'

    def draw(self, context):
        layout = self.layout
        
        scene = context.scene
        quadwild_config = scene.retopo.ops.quadwild.config
        
        col = layout.column(align=True)
        row = col.row(align=True)
        subcol = row.column(align=True)
        subcol.alignment = 'LEFT'
        subcol.prop(quadwild_config, "use_preprocess_simplify")
        subcol.prop(quadwild_config, "use_preprocess_subdivide")
        subcol = row.column(align=True)
        subcol.prop(quadwild_config, "planar_angle", text="Angle")
        subcol.prop(quadwild_config, "subdivide_iterations", text="Levels")


class RETOPO_PT_RetopologyEdgeOptionsPanel(bpy.types.Panel):
    bl_label = "Edge preservation"
    bl_description = "Edge preservation options"
    bl_idname = "RETOPO_PT_RetopologyEdgeOptionsPanel"
    bl_space_type = 'CONSOLE'
    bl_region_type = 'WINDOW'

    def draw(self, context):
        layout = self.layout
        
        scene = context.scene
        quadwild_config = scene.retopo.ops.quadwild.config
        
        col = layout.column(align=True)
        
        def draw_edge_option(value, text, prop_name=''):
            row = col.row(align=False)
            is_active = (value in quadwild_config.sharp_conditions)
            icon = ('CHECKBOX_HLT' if is_active else 'CHECKBOX_DEHLT')
            subrow = row.row(align=False)
            subrow.emboss = 'NONE'
            subrow.prop_enum(quadwild_config, "sharp_conditions", value, text='', icon=icon)
            if prop_name:
                row.prop(quadwild_config, prop_name, text=text)
            else:
                row.label(text=text)
        
        draw_edge_option('NORMAL', "Normal", "normals_angle")
        draw_edge_option('ANGLE', "Angle", "sharp_angle")
        draw_edge_option('SHARP', "Sharp")
        draw_edge_option('SEAM', "Seam")
        draw_edge_option('MATERIAL', "Material")
        draw_edge_option('UV', "UVs")


class RETOPO_PT_RetopologyAdvancedOptionsPanel(bpy.types.Panel):
    bl_label = "Advanced options"
    bl_description = "Advanced QuadWild options"
    bl_idname = "RETOPO_PT_RetopologyAdvancedOptionsPanel"
    bl_space_type = 'CONSOLE'
    bl_region_type = 'WINDOW'

    def draw(self, context):
        layout = self.layout
        
        scene = context.scene
        quadwild_config = scene.retopo.ops.quadwild.config
        
        col = layout.column(align=True)
        col.label(text="Polygon regulaity:")
        col.prop(quadwild_config, "quad_regularity", text="Quads")
        col.prop(quadwild_config, "non_quad_regularity", text="Non-quads")
        
        row = layout.row(align=True)
        col = row.column(align=True)
        col.alignment = 'LEFT'
        col.label(text="ILP method:")
        col.label(text="ILP solve:")
        col = row.column(align=True)
        col.prop(quadwild_config, "ilp_method", text="")
        col.prop(quadwild_config, "full_solve", text="Full")
        
        layout.prop(quadwild_config, "singularity_alignment")
        
        col = layout.column(align=True)
        col.prop(quadwild_config, "lost_constraint_iterations")
        row = col.row(align=True)
        row.prop(quadwild_config, "lost_constraint_flags")
        
        layout.prop(quadwild_config, "hard_parity_constraint")
        
        layout.prop(quadwild_config, "fixed_chart_clusters")
        
        col_main = layout.column(align=True)
        col_main.prop(quadwild_config, "use_flow_solver")
        row = col_main.row(align=True)
        row.enabled = quadwild_config.use_flow_solver
        col = row.column(align=True)
        col.alignment = 'LEFT'
        col.label(text="Flow config:")
        col.label(text="Solver config:")
        col = row.column(align=True)
        col.prop(quadwild_config, "flow_config", text="")
        col.prop(quadwild_config, "flow_solver_config", text="")


classes = (
    RETOPO_PT_RetopologyPanel,
    RETOPO_PT_RetopologyPreprocessingOptionsPanel,
    RETOPO_PT_RetopologyEdgeOptionsPanel,
    RETOPO_PT_RetopologyAdvancedOptionsPanel,
)