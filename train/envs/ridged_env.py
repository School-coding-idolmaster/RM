from train.envs.flat_env import FlatEnv
import xml.etree.ElementTree as ET
import tempfile

# Environment with ridges in the ground
# The ridges for now are round, since in MJX cylinders can't collide with boxes yet

class RidgedEnv(FlatEnv):
    def __init__(self, xml_path, spread=2.0, radius=0.2, **kwargs):
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        for i in range(15):
            ET.SubElement(worldbody, "geom",
                            name=f"ridge_{i}",
                            pos=f"-{i * spread + 1.0} 0 0",
                            size=f"{radius} 10.0",
                            quat="0.7071067811865476 0.7071067811865476 0.0 0.0",
                            type="cylinder",
                            contype="0",
                            conaffinity="1",
                            rgba="0.7 0.5 0.3 1.0")

        with tempfile.NamedTemporaryFile(suffix=".xml") as new_xml:
            new_xml_path = new_xml.name
            tree.write(new_xml_path)
            super().__init__(new_xml_path, **kwargs)