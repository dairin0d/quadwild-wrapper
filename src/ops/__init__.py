import bpy
from . import quadwild



class types():
    class scene(bpy.types.PropertyGroup):
        quadwild: bpy.props.PointerProperty(type=quadwild.types.scene)


types_classes = (
    *quadwild.types_classes,
    types.scene,
)


classes = (
    *quadwild.classes,
)


def register():

    quadwild.register()


def unregister():

    quadwild.unregister()
