import pybullet as p
import time
import math
import pybullet_data

useGui = True

if (useGui):
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

#p.loadURDF("samurai.urdf")
p.loadURDF("r2d2.urdf", [3, 3, 1])
# p.loadURDF("bicycle/bike.urdf", [5, 0, 1])
# p.loadURDF("teddy_large.urdf", [5, 0, 1])


rayFrom = []
rayTo = []
rayIds = []

numRays = 100

rayLen = 5

rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = True
blindspot_angle_deg = 60 # Angle of blind spot behind the vehicle (0 = no blind spot)
ray_swept_angle_deg = 90-blindspot_angle_deg/2  
start_angle = -(ray_swept_angle_deg) * math.pi/180  
end_angle =  (180 + ray_swept_angle_deg) * math.pi/180 
angle_range = end_angle - start_angle

for i in range(numRays):
  angle = start_angle + angle_range * i / (numRays - 1)
  rayFrom.append([0, 0, 1])
  rayTo.append([
      rayLen * math.cos(angle),
      rayLen * math.sin(angle),
      1
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

if (not useGui):
  p.stopStateLogging(timingLog)