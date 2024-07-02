import bpy
from . import ops


class Scene(bpy.types.PropertyGroup):
    ops: bpy.props.PointerProperty(type=ops.types.scene)


classes = [
    *ops.types_classes,
    Scene,
]


def register():

    bpy.types.Scene.retopo = bpy.props.PointerProperty(type=Scene)

    ops.register()


def unregister():

    del bpy.types.Scene.retopo

    ops.unregister()
