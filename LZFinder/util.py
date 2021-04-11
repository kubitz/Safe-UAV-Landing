from enum import Enum

class Label(Enum):
    background=[0,0,0]
    person=[1,1,1]
    bike=[2,2,2]
    car=[3,3,3]
    drone=[4,4,4]
    boat=[5,5,5]
    animal=[6,6,6]
    obstacle=[7,7,7]
    construction=[8,8,8]
    vegetation=[9,9,9]
    road=[10,10,10]
    sky=[11,11,11]

class RiskLevel(Enum):
    VERY_HIGH=100
    HIGH=20
    MEDIUM=10
    LOW=5
    ZERO=0

risk_table = dict([
    (Label.background, RiskLevel.ZERO),
    (Label.person, RiskLevel.VERY_HIGH),
    (Label.bike, RiskLevel.VERY_HIGH),
    (Label.car, RiskLevel.HIGH),
    (Label.drone, RiskLevel.HIGH),
    (Label.boat, RiskLevel.HIGH),
    (Label.animal, RiskLevel.HIGH),
    (Label.obstacle, RiskLevel.MEDIUM),
    (Label.construction, RiskLevel.MEDIUM),
    (Label.vegetation, RiskLevel.MEDIUM),
    (Label.road, RiskLevel.MEDIUM),
    (Label.sky, RiskLevel.ZERO),
])

class DroneCamera():
    def __init__(self):
        self.sensor_size=(50,100)
        self.focal_length=152
        self.resolution=(720,360)

    def get_image_footprint(self,altitude_GL):
        """Returns footprint of image based on ground-level altitude and camera parameters

        Args:
            altitude_GL (meter): altitude in meter

        Returns:
            tuple: footprint in meters
        """
        ax=self.sensor_size[0]*(altitude_GL/focal_length)
        ay=self.sensor_size[1]*5(altitude_GL/focal_length)
        return (ax,ay)
    
    def meters_to_pix(self,dist_meter,altitude_GL):
        footprint=self.get_image_footprint(altitude_GL)
        return footprint