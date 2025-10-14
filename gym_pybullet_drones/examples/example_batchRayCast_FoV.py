import pybullet as p
import time
import math
import pybullet_data

useGui = True

if useGui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

# Load environment
p.loadURDF("r2d2.urdf", [3, 3, 1])

# Parameters
numSensors = 15         # number of sensors around the circle
raysPerSensor = 13      # rays per sensor
fov = math.radians(20)  # FoV per sensor in radians
rayLen = 5              # ray length
sensorRadius = 1      # distance of sensors from origin

rayFrom = []
rayTo = []
rayIds = []

rayHitColor = [1, 0, 0]
rayMissColor = [0, 0, 1]
replaceLines = True

# Generate sensors on a circle
for s in range(numSensors):
    # Sensor position on circle
    sensorAngle = 2 * math.pi * s / numSensors
    sx = sensorRadius * math.cos(sensorAngle)
    sy = sensorRadius * math.sin(sensorAngle)
    sz = 1.0
    sensorPos = [sx, sy, sz]

    # Outward-facing yaw (points away from circle center)
    sensorYaw = sensorAngle  

    # Rays within FoV (spread around outward direction)
    for r in range(raysPerSensor):
        # evenly distribute rays within FoV
        offset = -fov / 2 + fov * (r / (raysPerSensor - 1))
        rayAngle = sensorYaw + offset

        dx = rayLen * math.cos(rayAngle)
        dy = rayLen * math.sin(rayAngle)
        dz = 0

        fromPoint = sensorPos
        toPoint = [sx + dx, sy + dy, sz + dz]

        rayFrom.append(fromPoint)
        rayTo.append(toPoint)

        if replaceLines:
            rayIds.append(p.addUserDebugLine(fromPoint, toPoint, rayMissColor))
        else:
            rayIds.append(-1)

if not useGui:
    timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

numSteps = 327680 if useGui else 10

for i in range(numSteps):
    p.stepSimulation()
    results = p.rayTestBatch(rayFrom, rayTo)

    if useGui:
        if not replaceLines:
            p.removeAllUserDebugItems()

        for j in range(len(rayFrom)):
            hitObjectUid = results[j][0]
            if hitObjectUid < 0:
                p.addUserDebugLine(rayFrom[j], rayTo[j], rayMissColor, replaceItemUniqueId=rayIds[j])
            else:
                hitPosition = results[j][3]
                p.addUserDebugLine(rayFrom[j], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[j])

if not useGui:
    p.stopStateLogging(timingLog)
