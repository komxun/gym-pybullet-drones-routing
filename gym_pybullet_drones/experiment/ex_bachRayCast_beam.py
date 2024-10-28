import pybullet as p
import time
import math
import pybullet_data
import numpy as np

useGui = True

if (useGui):
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER,0)
p.setGravity(0,0,-9.81)
p.setRealTimeSimulation(1) 


p.loadURDF("samurai.urdf")
r2d2id = p.loadURDF("r2d2.urdf", [3, 3, 1])

rayFrom = []
rayTo = []
rayIds = []

numRays = 100
rayLen = 5

rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = True

# Parameters for the beam
beam_angle = math.radians(45)  # Half-angle of the beam in radians
pitch_angle = math.radians(0)  # 30 degrees pitch
yaw_angle = math.radians(0)  # 45 degrees yaw

# Generate points on a sphere
indices = np.arange(0, numRays, dtype=float) + 0.5
phi = np.arccos(1 - 2*indices/numRays)
theta = np.pi * (1 + 5**0.5) * indices

x, y, z = rayLen * np.cos(theta) * np.sin(phi), rayLen * np.sin(theta) * np.sin(phi), rayLen * np.cos(phi)

# Apply pitch and yaw to filter points within the beam
def rotation_matrix(pitch, yaw):
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    
    pitch_matrix = np.array([
        [1, 0, 0],
        [0, cos_pitch, -sin_pitch],
        [0, sin_pitch, cos_pitch]
    ])
    
    yaw_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    
    return yaw_matrix @ pitch_matrix

rotation_mat = rotation_matrix(pitch_angle, yaw_angle)

# Rotate points
points = np.vstack((x, y, z)).T
rotated_points = points @ rotation_mat.T

# Select points within the beam angle
beam_center = np.array([0, 0, rayLen])
beam_points = []
for pt in rotated_points:
    if np.arccos(np.dot(pt, beam_center) / (np.linalg.norm(pt) * np.linalg.norm(beam_center))) <= beam_angle:
        beam_points.append(pt.tolist())

# Prepare rayFrom and rayTo
rayFrom = [[0, 0, 1] for _ in range(len(beam_points))]
rayTo = beam_points
rayIds = [p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor) for i in range(len(beam_points))]

if not useGui:
    timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

numSteps = 10
if useGui:
    numSteps = 327680

for i in range(numSteps):
    p.stepSimulation()
    for j in range(8):
        results = p.rayTestBatch(rayFrom, rayTo, j + 1)

    if useGui:
        if not replaceLines:
            p.removeAllUserDebugItems()

        for i in range(len(beam_points)):
            hitObjectUid = results[i][0]

            if hitObjectUid < 0:
                hitPosition = [0, 0, 0]
                p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
            else:
                hitPosition = results[i][3]
                p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])

kuy = [results[x] for x in range(len(results)) if results[x][0] != -1]
if not useGui:
    p.stopStateLogging(timingLog)