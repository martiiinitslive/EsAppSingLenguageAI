"""
Wrapper para mantener compatibilidad: delega en convert_mediapipe_to_blender.py
para convertir poses_mediapipe_video.json a poses_converted_video.json usando
el mismo pipeline (vectores de reposo, ejes Blender y nombres de hueso MakeHuman).
"""

from convert_mediapipe_to_blender import main as convert_main


def main():
    convert_main()


if __name__ == "__main__":
    main()
