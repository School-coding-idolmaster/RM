from train.envs.flat_env import FlatEnv
import xml.etree.ElementTree as ET
import tempfile

# Environment with very little friction, modeled after frozen_lake from RoboGrammar

class FrozenLakeEnv(FlatEnv):
    def __init__(self, xml_path, **kwargs):
        tree = ET.parse(xml_path)

        floor = tree.findall(".//geom[@type='plane']")
        
        if len(floor) != 1:
            raise ValueError(f"Expected exactly 1 ground plane in XML, found {len(floor)}")

        floor[0].set("friction", "0.05") # This was coefficient used in RoboGrammar

        with tempfile.NamedTemporaryFile(suffix=".xml") as new_xml:
            new_xml_path = new_xml.name
            tree.write(new_xml_path)
            super().__init__(new_xml_path, **kwargs)