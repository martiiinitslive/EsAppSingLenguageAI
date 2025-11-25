import bpy
import json
import os
from mathutils import Quaternion, Euler

BASE_DIR = os.path.dirname(__file__)
POSES_JSON_PATH = os.path.join(BASE_DIR, "poses_converted.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "rendered_poses")
ARMATURE_NAME = "Human.rig"

def load_poses(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No encontrado: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def apply_pose_to_bones(armature, pose_data):
    """Aplica rotaciones de pose a los huesos"""
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    bones_data = pose_data.get('bones', {})
    applied_count = 0
    
    for bone_name, bone_data in bones_data.items():
        pb = armature.pose.bones.get(bone_name)
        if pb is None:
            continue
        
        if 'rot_quat' in bone_data and bone_data['rot_quat']:
            q = bone_data['rot_quat']
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = Quaternion((q[0], q[1], q[2], q[3]))
            applied_count += 1
    
    bpy.ops.object.mode_set(mode='OBJECT')
    return applied_count

def render_frame(output_path):
    """Renderiza el frame actual"""
    scene = bpy.context.scene
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    bpy.ops.render.render(write_still=True)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Cargar poses
    data = load_poses(POSES_JSON_PATH)
    poses = data.get('poses', {})
    print(f"Total poses cargadas: {len(poses)}")
    
    # Encontrar armature
    armature = bpy.data.objects.get(ARMATURE_NAME)
    if not armature:
        raise RuntimeError(f"Armature '{ARMATURE_NAME}' no encontrado")
    
    print(f"✓ Armature encontrado: {armature.name}\n")
    
    # ✅ RENDERIZAR VARIAS POSES
    poses_to_render = ['A', 'B', 'C', 'D', 'E', 'L']  # Prueba estas
    
    for pose_key in poses_to_render:
        if pose_key not in poses:
            print(f"✗ Pose {pose_key} no existe")
            continue
        
        print(f"[RENDER] Pose: {pose_key}")
        pose_data = poses[pose_key]
        applied = apply_pose_to_bones(armature, pose_data)
        print(f"  ✓ Aplicados {applied} huesos")
        
        output_file = os.path.join(OUTPUT_DIR, f"{pose_key}.png")
        render_frame(output_file)
        print(f"  ✓ Guardado: {output_file}\n")
    
    print("✓ DONE - Renderizado completo!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise