import argparse
import logging
import re
import sys
from io import StringIO
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class Body:
    def __init__(self, size, leg=None):
        self.size = size
        self.leg = leg

    def __str__(self):
        if self.leg:
            return f"Body: {self.size}m, ({', '.join(map(str, self.leg))})"
        return f"Body: {self.size}m"


class Roll:
    def __init__(self):
        pass

    def __str__(self):
        return "Roll"


class Twist:
    def __init__(self):
        pass

    def __str__(self):
        return "Twist"


class Rigid:
    def __init__(self):
        pass

    def __str__(self):
        return "Rigid"


class Knee:
    def __init__(self, degree):
        self.degree = degree

    def __str__(self):
        return f"Knee: {self.degree}°"


class Elbow:
    def __init__(self, degree):
        self.degree = degree

    def __str__(self):
        return f"Elbow: {self.degree}°"


class Mount:
    def __init__(self):
        pass

    def __str__(self):
        return "Mount"


class Limb:
    def __init__(self, size):
        self.size = size

    def __str__(self):
        return f"Limb: {self.size}m"


class Wheel:
    def __init__(self):
        pass

    def __str__(self):
        return "Wheel"


def parse_float_till_non_number(text):
    """Parse the first float from a string."""
    match = re.search(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass
    return None


def tokenize(src: str):
    """Split a component graph string into tokens: [string], (, or )."""
    tokens = []
    in_brackets = False
    current = ""
    for char in src:
        if in_brackets:
            if char == "]":
                tokens.append(current + "]")
                in_brackets = False
            else:
                current += char
        elif char == "[":
            in_brackets = True
            current = "["
        elif char == "(":
            tokens.append("(")
        elif char == ")":
            tokens.append(")")
    return tokens


def compile_robot(src: str):
    """Parse a component graph string into a list of robot part objects."""
    in_leg = False
    robot = []
    for token in tokenize(src):
        if token == "(":
            in_leg = True
            leg = []
        elif token == ")":
            if not in_leg:
                raise ValueError("Unexpected ')' outside of leg definition")
            in_leg = False
            robot[-1].leg = leg
        elif in_leg:
            if token.startswith("[knee"):
                leg.append(Knee(parse_float_till_non_number(token.split("=")[1][:-3])))
            elif token.startswith("[limb"):
                leg.append(
                    Limb(parse_float_till_non_number(token.split("=")[1][:-2]) / 100)
                )
            elif token.startswith("[elbow"):
                leg.append(Elbow(parse_float_till_non_number(token.split("=")[1][:-3])))
            elif token.startswith("[wheel"):
                leg.append(Wheel())
            elif token.startswith("[roll"):
                leg.append(Roll())
            elif token.startswith("[rigid"):
                leg.append(Rigid())
            elif token.startswith("[mount"):
                leg.append(Mount())
        else:
            if token.startswith("[body"):
                if "=" in token:
                    size = parse_float_till_non_number(token.split("=")[1][:-2]) / 100
                else:
                    size = 0.15
                robot.append(Body(size))
            elif token.startswith("[roll"):
                robot.append(Roll())
            elif token.startswith("[twist"):
                robot.append(Twist())
            elif token.startswith("[rigid"):
                robot.append(Rigid())
            elif token.startswith("[wheel"):
                robot.append(Wheel())
            elif token.startswith("[mount"):
                robot.append(Mount())
    return robot


class _BuildState:
    def __init__(self):
        self.joint_no = 0
        self.motors = []


def make_leg(leg, string_builder, reverse, state):
    """Build XML for a leg, starting at (0,0,0) and extending along x."""
    x_pos = 0
    body_count = 0
    for part in leg:
        if isinstance(part, Limb):
            size = part.size
            string_builder.write(
                f'<geom type="capsule" size="0.02" fromto="{x_pos} 0 0 {x_pos + size} 0 0" rgba="0.2 0.6 0.8 1"/>',
            )
            x_pos += size
        elif isinstance(part, Mount):
            size = 0.1
            string_builder.write(
                f'<geom type="capsule" size="0.02" fromto="{x_pos} 0 0 {x_pos + size} 0 0" rgba="0.2 0.6 0.8 1"/>',
            )
            x_pos += size
        elif isinstance(part, Roll):
            string_builder.write(f"<body>")
            body_count += 1
            axis = "0 0 1" if not reverse else "0 0 -1"
            string_builder.write(
                f'<site type="cylinder" size="0.02" fromto="{x_pos} 0 0.02 {x_pos} 0 -0.02" rgba="0.15 0.55 0.15 1"/>'
            )
            string_builder.write(
                f'<joint name="j{state.joint_no}" pos="{x_pos} 0 0" axis="{axis}" range="-90 90" damping="0.1"/>'
            )
            state.joint_no += 1
        elif isinstance(part, Knee):
            string_builder.write(f"<body>")
            body_count += 1
            ref = part.degree - 30
            string_builder.write(
                f'<joint name="j{state.joint_no}" pos="{x_pos} 0 0" axis="0 1 0" range="0 90" ref="{ref}"/>'
            )
            string_builder.write(
                f'<geom type="cylinder" size=".02" fromto="{x_pos} -0.02 0 {x_pos} 0.02 0" rgba="1 0.5 0.3 1"/>'
            )
            state.joint_no += 1
        elif isinstance(part, Elbow):
            string_builder.write(f"<body pos='{x_pos} 0 0' quat='0.70710678118654750 0.0 0.70710678118654750 0.0'>")
            x_pos = 0
            body_count += 1
            string_builder.write(
                f'<site type="cylinder" size=".021" fromto="{x_pos} 0 -0.02 {x_pos} 0 0.02" rgba="1 0.5 0.3 1"/>'
            )
            ref = part.degree
            axis = "0 0 -1" if not reverse else "0 0 1"
            string_builder.write(
                f'<joint name="j{state.joint_no}" pos="{x_pos} 0 0" axis="{axis}" range="0 180" ref="{ref}" damping="0.1"/>'
            )
            state.joint_no += 1
        elif isinstance(part, Rigid):
            size = 0.021
            string_builder.write(
                f'<geom type="box" size="{size} {size} {size}" pos="{x_pos} 0 0" rgba="1 0.5 0.3 1"/>'
            )
        elif isinstance(part, Wheel):
            x_pos += 0.01
            string_builder.write(f'<body pos="{x_pos} 0 0" quat="0.70710678118654750 0.0 0.70710678118654750 0.0">')
            string_builder.write(f'<joint name="j{state.joint_no}"/>')
            state.motors.append(state.joint_no)
            state.joint_no += 1
            string_builder.write(
                f'<geom type="cylinder" size=".05 .01" rgba=".5 .5 1 1"/>'
            )
            string_builder.write("</body>")
        else:
            raise ValueError(f"Unknown leg part: {part}")
    for _ in range(body_count):
        string_builder.write("</body>")


def make_body(start_x, end_x, string_builder, state, leg=None, body_count=0):
    string_builder.write(
        f'<geom pos="0 0 0" size="0.04" type="capsule" fromto="{start_x} 0 0 {end_x} 0 0" rgba="0.2 0.6 0.8 1"/>'
    )
    if body_count == 0:
        string_builder.write(
            f'<joint armature="0" damping="0" limited="false" margin="0.01" pos="0 0 0" type="free"/>'
        )
    if leg:
        middle = (start_x + end_x) / 2
        string_builder.write(f'<body pos="{middle} 0.02 0" quat="0.7071067811865475 0.0 0.0 0.7071067811865475">')
        make_leg(leg, string_builder, False, state)
        string_builder.write("</body>")
        string_builder.write(f'<body pos="{middle} -0.02 0" quat="0.7071067811865475 0.0 0.0 -0.7071067811865475">')
        make_leg(leg, string_builder, True, state)
        string_builder.write("</body>")


def make_prefix(string_builder):
    string_builder.write(
        """
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option timestep="0.01" iterations="4" />
  <custom>
    <!-- brax custom params -->
    <numeric data="1000" name="constraint_limit_stiffness"/>
    <numeric data="4000" name="constraint_stiffness"/>
    <numeric data="10" name="constraint_ang_damping"/>
    <numeric data="20" name="constraint_vel_damping"/>
    <numeric data="0.5" name="joint_scale_pos"/>
    <numeric data="0.2" name="joint_scale_ang"/>
    <numeric data="0.0" name="ang_damping"/>
    <numeric data="1" name="spring_mass_scale"/>
    <numeric data="1" name="spring_inertia_scale"/>
    <numeric data="15" name="solver_maxls"/>
  </custom>
  <default>
		<joint armature="1" />
		<geom contype="1" conaffinity="0" condim="3" />
	</default>
  <asset>
		<texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4" />
		<material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2" />
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom type="plane" size="3 3 .01" material="grid" conaffinity="1" contype="0" />
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
"""
    )


def make_suffix(string_builder, state):
    string_builder.write(
        "</body></worldbody>",
    )
    if state.joint_no > 0:
        string_builder.write("""<actuator>""")
        for i in range(state.joint_no):
            if i not in state.motors:
                string_builder.write(f'<motor joint="j{i}" ctrllimited="true" ctrlrange="-1.0 1.0" gear="50"/>')
        string_builder.write("""</actuator>""")
    string_builder.write("</mujoco>")


def make_robot(robot):
    """Convert a compiled parts list into a MuJoCo XML string."""
    state = _BuildState()
    string_builder = StringIO()
    make_prefix(string_builder)
    start_x = 0
    body_count = 0
    for k, main_part in enumerate(robot):
        if isinstance(main_part, Body):
            make_body(
                start_x,
                start_x + main_part.size,
                string_builder,
                state,
                main_part.leg,
                body_count
            )
            start_x += main_part.size
        elif isinstance(main_part, Mount):
            size = 0.04
            string_builder.write(
                f'<geom type="cylinder" size="0.02" fromto="{start_x} 0 0 {start_x + size} 0 0" rgba="0.2 0.6 0.8 1"/>',
            )
            start_x += size
        elif isinstance(main_part, Roll):
            string_builder.write("<body>")
            string_builder.write(
                f'<geom type="cylinder" size=".042" fromto="{start_x} 0 0.04 {start_x} 0 -0.04" rgba="0.15 0.55 0.15 1"/>'
            )
            string_builder.write(
                f'<joint name="j{state.joint_no}" pos="{start_x} 0.0 0" axis="0 0 1" range="-90 90" />'
            )
            state.joint_no += 1
            body_count += 1
        elif isinstance(main_part, Twist):
            string_builder.write("<body>")
            string_builder.write(
                f'<site type="cylinder" size=".042" fromto="{start_x-0.04} 0 0 {start_x+0.04} 0 0" rgba="0.15 0.55 0.15 1"/>'
            )
            string_builder.write(
                f'<joint name="j{state.joint_no}" pos="{start_x} 0.0 0" axis="1 0 0" range="-90 90" damping="0.1"/>'
            )
            state.joint_no += 1
            body_count += 1
        elif isinstance(main_part, Rigid):
            string_builder.write("<body>")
            string_builder.write(
                f'<site type="box" size=".042 .042 .042" fromto="{start_x-0.04} 0 0 {start_x+0.04} 0 0 " rgba="1 0.5 0.3 1"/>'
            )
            body_count += 1
        elif isinstance(main_part, Wheel) and k == len(robot) - 1:
            start_x += 0.01
            string_builder.write(f'<body pos="{start_x} 0 0" quat="0.7071067811865475 0.0 0.7071067811865475 0.0">')
            string_builder.write(
                f'<geom type="cylinder" size=".05 .01" rgba=".5 .5 1 1"/>'
            )
            string_builder.write("</body>")
        elif isinstance(main_part, Wheel) and k == 0:
            string_builder.write(f'<body pos="{start_x - 0.01} 0 0" quat="0.70710678118654750 0.0 0.70710678118654750 0.0">')
            string_builder.write(
                f'<geom type="cylinder" size=".05 .01" rgba=".5 .5 1 1"/>'
            )
            string_builder.write("</body>")
        else:
            raise ValueError(f"Unknown part: {main_part}")
    for _ in range(body_count):
        string_builder.write("</body>")
    make_suffix(string_builder, state)

    # Parse and indent
    root = ET.fromstring(string_builder.getvalue())
    ET.indent(root)

    return ET.tostring(root, encoding="unicode")


def xml_has_joints(xml_file_path):
    """Check if a compiled robot XML contains actuator joints."""
    try:
        tree = ET.parse(xml_file_path)
        actuator = tree.find('.//actuator')
        return actuator is not None and len(actuator) > 0
    except (ET.ParseError, OSError) as e:
        logger.error(f"Failed to parse XML {xml_file_path}: {e}")
        return False


def process_input(input_stream):
    """Read all lines from input and join them into a single string."""
    return "".join(line.rstrip() for line in input_stream)


def main():
    """Compile a robot component graph from stdin/file into a MuJoCo XML file."""
    parser = argparse.ArgumentParser(
        description="Compile a robot description into a Mujoco XML file."
    )
    parser.add_argument(
        "-i", "--input", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument("-o", "--output", type=str, help="The output file.")

    args = parser.parse_args()
    robot_xml_string = make_robot(compile_robot(process_input(args.input)))
    if args.output:
        with open(args.output, "w") as f:
            f.write(robot_xml_string)
    else:
        print(robot_xml_string)


if __name__ == "__main__":
    main()
