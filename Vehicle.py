def get_bb_area(bb):
    """Calculates the area of the bounding box

    :param bb: list of length 4 in the format: [min x, min y, max x, max y]
    :return: int, the area of the bounding box
    """
    x1 = bb[0]
    y1 = bb[1]
    x2 = bb[2]
    y2 = bb[3]

    width = x2 - x1
    height = y2 - y1

    return width * height


def get_co2(vehicle_class):
    """Returns the raw gco2/km for the class of vehicle identified
    """
    # [2, 3, 5, 7] = [car, motorbike, bus and truck]
    if vehicle_class == 2:
        # car
        return 166
    elif vehicle_class == 3:
        # motor bike
        return 117
    elif vehicle_class == 5:
        # bus
        return 1215
    elif vehicle_class == 7:
        # 'truck'
        return 226


class Vehicle:
    def __init__(self, track_id, vehicle_class, bb, status):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.co2 = get_co2(self.vehicle_class)
        self.bb = bb
        self.bb_area = get_bb_area(bb)
        self.status = status
        self.confirmed = False
        self.distances = []

    def set_bb(self, bb):
        """Updates the bb and area of the vehicle

        :param bb: list of length 4 in the format: [min x, min y, max x, max y]
        """
        self.bb = bb
        self.bb_area = get_bb_area(bb)

    def get_average_distance(self):
        """Calculates the average distance from the vehicle

        CURRENTLY JUST TAKES RAW LIST, NO ACCOUNT FOR INCORRECT RESULTS
        :return: float, average distance
        """
        return sum(self.distances)/len(self.distances)

    def __str__(self):
        """Override of string representation

        :return: str, human-readable string output of the vehicle class
        """
        return 'id: {}, \nclass: {}, \nco2: {}, \nbb: {}, \nbb area: {}, \nstatus: {}, \nconfirmed: {}, \ndistances: {}'\
            .format(self.track_id, self.vehicle_class, self.co2, self.bb, self.bb_area, self.status, self.confirmed, self.distances)