# Pipeline
## Depth Completition

Depth images are in mm and the maximum value is 3.2 meters. 
Parameters: 
- Max Depth is kept as 5 meters
- Diamond Kernal of size 7 is used
- Kernel5, kernal7 and *kernel17* is used instead of kernal31. 

The reason for choosing kernal17 was that it was less distorting the depth map (although it didnt matter on the object)