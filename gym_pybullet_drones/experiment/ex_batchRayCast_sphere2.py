import pybullet as p
import time
import math
import pybullet_data
import numpy as np

def fibonacci_sphere(samples=1000, dist=1):

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([dist*x, dist*y, dist*z])

    return points

useGui = True

if (useGui):
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

#p.loadURDF("samurai.urdf")
r2d2id = p.loadURDF("r2d2.urdf", [3, 3, 1])

rayFrom = []
rayTo = []
rayIds = []

numRays = 40

rayLen = 5

rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = True


rayTo = fibonacci_sphere(numRays, rayLen)

rayFrom = [[0,0,1] for _ in range(numRays)]
# rayTo = [[x[i], y[i], z[i]] for i in range(numRays)]
rayIds = [p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor) for i in range(numRays)]

if (not useGui):
  timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

numSteps = 10
if (useGui):
  numSteps = 327680

for i in range(numSteps):
  p.stepSimulation()
  for j in range(8):
    results = p.rayTestBatch(rayFrom, rayTo, j + 1)

  #for i in range (10):
  #	p.removeAllUserDebugItems()

  if (useGui):
    if (not replaceLines):
      p.removeAllUserDebugItems()

    for i in range(numRays):
      hitObjectUid = results[i][0]

      if (hitObjectUid < 0):
        hitPosition = [0, 0, 0]
        p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
      else:
        hitPosition = results[i][3]
        p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])

  #time.sleep(1./240.)
kuy = [results[x] for x in range(len(results)) if results[x][0]  != -1]
if (not useGui):
  p.stopStateLogging(timingLog)