import bpy
from bpy.utils import register_class, unregister_class
from . import ops, ui, types, preferences


classes = (
    *types.classes,
    *preferences.classes,
    *ops.classes,
    *ui.classes,
)


def register():

    for cls in classes:
        register_class(cls)

    types.register()


def unregister():

    for cls in reversed(classes):
        unregister_class(cls)

    types.unregister()