import bpy

pos_range = 3
fov=50
pi=3.14159
scene = bpy.context.scene
#scene.render.image_settings.file_fomat = 'jpg'
scene.render.resolution_x = 512
scene.render.resolution_y = 512

scene.camera.data.angle = fov*(pi/180.0)



for object in bpy.data.objects:
    if object.name != 'Lamp' and object.name!= 'Camera':
        object.select = True
bpy.ops.object.delete()
bpy.ops.mesh.primitive_ico_sphere_add()
bpy.types.ImageFormatSettings.color_mode='BW'

for xx in range(-pos_range,pos_range+1):
    for yy in range(-pos_range,pos_range+1):
        for zz in range(-pos_range,pos_range+1):
            bpy.data.objects['Icosphere'].location = (xx,yy,zz)
            #path = '/Users/grndfthrprdx/Development/bpy_data/sphere_'+str(xx)+'_'+str(yy)+'_'+str(zz)+'.jpg'
            path = 'sphere_'+str(xx)+'_'+str(yy)+'_'+str(zz)+'.jpg'
            bpy.data.scenes['Scene'].render.filepath = path
            bpy.ops.render.render(write_still=True)
