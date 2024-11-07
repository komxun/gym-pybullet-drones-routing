import pybullet as p
import time
import math
import pybullet_data
import pkg_resources

useGui = True

if (useGui):
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0,0,-9.81)

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(rgbBackground=[1, 1, 1])
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
# p.loadURDF("plane.urdf")
p.loadURDF("sphere2.urdf", [1.5, 1.5, 0.5])
# p.loadURDF("bicycle/bike.urdf", [5, 0, 1])
# p.loadURDF("teddy_large.urdf", [5, 0, 1])
p.setRealTimeSimulation(1) 

rayFrom = []
rayTo = []
rayIds = []

numRays = 200
rayLen = 3

rayHitColor = [1, 0, 0]
rayMissColor = [0, 0, 0.5]

replaceLines = True

for i in range(numRays):
  rayFrom.append([0, 0, 0.1])
  rayTo.append([
      rayLen * math.sin(2. * math.pi * float(i) / numRays),
      rayLen * math.cos(2. * math.pi * float(i) / numRays), 1
  ])
  if (replaceLines):
    rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
  else:
    rayIds.append(-1)


if (not useGui):
  timingLog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "rayCastBench.json")

numSteps = 10
if (useGui):
  numSteps = 327680

for i in range(numSteps):
  p.stepSimulation()
  for j in range(1):
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
        p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i], lineWidth=2)
      else:
        hitPosition = results[i][3]
        p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i], lineWidth=2)

  #time.sleep(1./240.)

if (not useGui):
  p.stopStateLogging(timingLog)