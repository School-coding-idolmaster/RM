"""Render a recorded Hyblock MPC trajectory (record_episode.py output) to an mp4.

Replays qpos with mj_kinematics only (no simulation), so the video shows exactly the recorded episode.
Render-only changes to the model: lighter floor texture, larger visual floor (the plane collides everywhere anyway),
sky gradient. usage: python render_video.py traj.npz out.mp4 --speed 2 --title "..."
"""
import argparse
import json
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument("traj")
p.add_argument("out")
p.add_argument("--speed", type=float, default=2.0)
p.add_argument("--fps", type=int, default=30)
p.add_argument("--w", type=int, default=1280)
p.add_argument("--h", type=int, default=720)
p.add_argument("--title", default="")
p.add_argument("--frames", default="", help="comma list of sim times [s] to save as png instead of a video")
a = p.parse_args()

z = np.load(a.traj)
meta = json.load(open(a.traj.replace(".npz", ".json")))
qpos, rewards, xml = z["qpos"], z["rewards"], str(z["xml"])
dt = meta["dt_env"]

import re
# --- render-only look (the trajectory is replayed, nothing is simulated) ---
# floor: near-white with thin 0.5 m grid lines; much larger visual plane
xml = re.sub(r'<texture name="grid"[^>]*/>',
             '<texture name="grid" type="2d" builtin="checker" mark="edge" width="512" height="512" '
             'rgb1=".97 .97 .98" rgb2=".97 .97 .98" markrgb=".80 .82 .86" />', xml)
xml = xml.replace('texrepeat="1 1" texuniform="true" reflectance=".2"', 'texrepeat="2 2" texuniform="true" reflectance="0" emission=".5"')
xml = xml.replace('size="20 20 .01"', 'size="80 80 .01"')
xml = xml.replace("<asset>", '<asset>\n    <texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2=".93 .95 .98" width="256" height="256" />', 1)
xml = xml.replace("<worldbody>", '<visual><headlight ambient=".45 .45 .45" diffuse=".3 .3 .3" specular="0 0 0"/>'
                  '<quality shadowsize="4096" offsamples="8"/><global offwidth="1920" offheight="1080"/>'
                  '</visual>\n  <worldbody>', 1)
# one soft key light from above-front that casts shadows (replaces the flat overhead light)
xml = re.sub(r'<light directional="true"[^>]*/>',
             '<light mode="trackcom" directional="true" castshadow="false" diffuse=".7 .7 .7" specular=".05 .05 .05" pos="1 -1 4" dir="-.25 .25 -1" />', xml)
# colours as in the slides: light block, navy end plates, steel-blue hub, dark feet
COL = {"foot": ".15 .15 .17 1", "block": ".95 .95 .96 1", "P1": ".12 .18 .55 1", "P2": ".12 .18 .55 1", "hub": ".45 .55 .72 1"}
def _recolour(mt):
    name, rest = mt.group(1), mt.group(2)
    key = next((k for k in COL if f"_{k}" in name), None)
    if key is None:
        return mt.group(0)
    return f'<geom name="{name}"' + re.sub(r'rgba="[^"]+"', f'rgba="{COL[key]}"', rest)
xml = re.sub(r'<geom name="([^"]+)"([^>]*>)', _recolour, xml)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, a.h, a.w)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.azimuth, cam.elevation, cam.distance = -112.0, -20.0, 2.2
opt = mujoco.MjvOption()

x0 = qpos[0, 0]
dist = np.concatenate([[0.0], np.cumsum(rewards) * dt])  # forward distance after each env step (-x)
# smooth camera target so the view does not shake with the body
look = qpos[:, :3].copy()
k = 30
pad = np.pad(look, ((k, k), (0, 0)), mode="edge")
look_s = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for i in range(3)], 1)

try:
    font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
except OSError:
    font_b = font = ImageFont.load_default()


def frame(i):
    d.qpos[:] = qpos[i]
    mujoco.mj_kinematics(m, d)
    cam.lookat[:] = [look_s[i, 0], look_s[i, 1], 0.18]
    r.update_scene(d, cam, opt)
    img = Image.fromarray(r.render()).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    g = ImageDraw.Draw(ov)
    t = i * dt
    line2 = f"t = {t:4.1f} s    distance = {dist[i]:4.1f} m" + (f"    {a.speed:g}x speed" if a.speed != 1 else "")
    wbox = max(g.textlength(a.title, font=font_b) if a.title else 0, g.textlength(line2, font=font)) + 48
    g.rounded_rectangle((24, 20, 24 + wbox, 20 + (98 if a.title else 56)), radius=12, fill=(255, 255, 255, 215))
    y = 32
    if a.title:
        g.text((48, y), a.title, font=font_b, fill=(10, 20, 90)); y += 44
    g.text((48, y), line2, font=font, fill=(60, 60, 60))
    img = Image.alpha_composite(img, ov).convert("RGB")
    return img


n = len(qpos)
if a.frames:
    for ts in [float(x) for x in a.frames.split(",")]:
        i = min(n - 1, int(round(ts / dt)))
        frame(i).save(f"{a.out}_{ts:05.1f}s.png")
    sys.exit()

T = (n - 1) * dt
nf = int(T / a.speed * a.fps) + 1
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{a.w}x{a.h}", "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                       "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
for f in range(nf):
    i = min(n - 1, int(round(f * a.speed / a.fps / dt)))
    ff.stdin.write(frame(i).tobytes())
# hold the last frame 1.5 s so the final distance can be read
last = frame(n - 1).tobytes()
for _ in range(int(1.5 * a.fps)):
    ff.stdin.write(last)
ff.stdin.close()
ff.wait()
print("wrote", a.out, nf, "frames")
