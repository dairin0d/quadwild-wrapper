import bpy


class quadwild(bpy.types.PropertyGroup):
    pass


def draw_quadwild(self, context, layout):
    layout = layout.column()
    
    scene = context.scene
    quadwild_config = scene.retopo.ops.quadwild.config
    quadwild_config.draw_installation_ui(layout)


class Retopo(bpy.types.AddonPreferences):
    bl_idname = "QuadWild wrapper"

    settings: bpy.props.EnumProperty(
        name='Settings',
        description='Settings to display',
        items=[
            ('QUADWILD', 'QuadWild', ''),
        ],
        default='QUADWILD')

    quadwild: bpy.props.PointerProperty(type=quadwild)

    def draw(self, context):
        layout = self.layout

        column = layout.column(align=True)
        row = column.row(align=True)
        row.prop(self, 'settings', expand=True)

        box = column.box()
        if self.settings == 'QUADWILD':
            draw_quadwild(self, context, box)


classes = (
    quadwild,
    Retopo,
)