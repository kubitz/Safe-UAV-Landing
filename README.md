# Safe-UAV-Landing (Python)
Automatic Safe Landing Zone Estimation for UAVs (Quadcopters) in unkown environments written in Python. 

# Risk Map Generation - Safe Drone Landing

![](https://s4.gifyu.com/images/firstVersion.gif)

The current approach to rank the best possible landing zones can be divided in three steps:

1. Candidate Landing Zone Proposal
2. Risk Map Generation
3. Candidate Zone Ranking based on custom cost function

(Note that 1. and 2. can happen in parallel)

## Candidate Zone Proposal 
This step aims to categorically reject any zone that is considered unsafe. This step uses only the output of the object detector. 

First, the image is fed through the object detector which aims to detect an obstacle that fits in the following categories:
* Vehicle (Car/Truck)
* Bicycle
* Pedestrian

Based on the output bounding box, a set **NO LANDING ZONE** are defined (e.g. 2 meters for pedestrians, 3 meters for cars) - see red circles in the picture here below. 

Depending on a stride parameter, the frame is filled with as many circles as possible (with a diameter corresponding to the min. landing zone dimensions). Any circle that interesect with a **NO LANDING ZONE** is disqualified. 

![](https://i.ibb.co/3zxn8NM/proposals.png)

## Risk Map Generation 
In parallel to landing zone proposal a risk map is generated based on the *semantic segmentation output only*.

Each output class labelled by the segmentation block is assigned a risk score, as shown in the table below, leading to a 2D array containing values between 0-100. 

| Category                    | Risk Level | Score |
|-----------------------------|------------|-------|
| Road                        | MEDIUM     | 20    |
| High Vegetation             | HIGH       | 50    |
| Pedestrian                  | VERY HIGH  | 100   |
| Car                         | VERY HIGH  | 100   |
| Low Vegetation (i.e. grass) | LOW        | 5     |
| Building/Roof               | HIGH       | 50    |
| Fence/Pole                  | HIGH       | 20    |
| Background                  | HIGH       | 0     |

This risk map is then smoothened by applying a 2D-Gaussian filter, and finally normalized, to have every single array cell correspond to a floating point number between 0.0 and 1.0. You can see an example of the smoothened risk map at the top of the file, in the left frame of GIF. 

## Cost Function and Landing Zone Ranking

Once the last two steps have been computed, each of the landing zone proposal is evaluated based on a custom cost function. 

At the present time this is based on two factor which can be weighted based on set parameters:

1. **The distance between the drone and proposed landing zone**

To calculate this, we assume that the position of the drone is equal to the center of the capture frame (i.e. the camera is strictly facing downwards and the drone is perfectly horizontal). 
The distance between the center of the proposed zone, and center of the image is computed and normalized (based on the furthest possible distance). This gives a number between 0.0 and 1.0.

2. **The average risk level in the proposed landing area**

This is pretty straight forward, all pixels value within the area of the potential landing zone are averaged. This gives a number between 0.0 and 1.0.

A weighted average is then done based on those two factors leading to a final score between 0 and 1 which is used for the ranking. 

## Alternative methods
Based on the previous meeting, it seems that another approach that could make the verification easier, would be to not discard any landing zone in step 1., but rather score them extremely low to make sure they get pushed back to the bottom of the ranking (maybe by assigning negative scores, and use a max(score, 0) at the end). 
