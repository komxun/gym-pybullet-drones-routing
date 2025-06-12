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
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

#p.loadURDF("samurai.urdf")
r2d2id = p.loadURDF("r2d2.urdf", [3, 3, 1])

rayFrom = []
rayTo = []
rayIds = []

numRays = 24

rayLen = 2

rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = False

# sunflower on a sphere: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/44164075#44164075
indices = np.arange(0, numRays, dtype=float) + 0.5


phi = np.arccos(1 - 2*indices/numRays)
theta = np.pi * (1 + 5**0.5) * indices

x, y, z = rayLen* np.cos(theta) * np.sin(phi), rayLen* np.sin(theta) * np.sin(phi), rayLen*np.cos(phi)
rayFrom = [[0,0,1] for _ in range(numRays)]
rayTo = [[x[i], y[i], z[i]] for i in range(numRays)]
rayIds = [p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor,lifeTime=0.01) for i in range(numRays)]

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
        p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor)
      else:
        hitPosition = results[i][3]
        p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor)

  #time.sleep(1./240.)
kuy = [results[x] for x in range(len(results)) if results[x][0]  != -1]
if (not useGui):
  p.stopStateLogging(timingLog)