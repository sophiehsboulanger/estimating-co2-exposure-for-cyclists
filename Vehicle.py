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
        # car 166
        return 5
    elif vehicle_class == 3:
        # motor bike 117
        return 3
    elif vehicle_class == 5:
        # bus 356
        return 10
    elif vehicle_class == 7:
        # 'truck' 226
        return 6


def divide_chunks(distances, divisor):
    for i in range(0, len(distances), divisor):
        yield distances[i:i + divisor]


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
        self.score = None

    def set_bb(self, bb):
        """Updates the bb and area of the vehicle

        :param bb: list of length 4 in the format: [min x, min y, max x, max y]
        """
        self.bb = bb
        self.bb_area = get_bb_area(bb)

    def add_distance(self, distance):
        self.distances.append(distance)

    def get_average_distance(self):
        """Calculates the average distance from the vehicle

        CURRENTLY JUST TAKES RAW LIST, NO ACCOUNT FOR INCORRECT RESULTS
        :return: float, average distance
        """
        return sum(self.distances) / len(self.distances)

    def get_score(self, divisor):
        # divide the distances into chunks based on the fps
        chunks = divide_chunks(self.distances, divisor)
        averages = []
        for chunk in chunks:
            # currently takes no account for outliers
            averages.append(sum(chunk) / len(chunk))
            # print(sum(chunk) / len(chunk))
        # get score based on distance for each second of video
        score = 0
        for average in averages:
            if average > 2000:
                # far enough to be considered background so no modifier
                score = score + self.co2
            elif 1000 < average <= 2000:
                # 30% modifier
                score = score + self.co2 * 1.3
            elif average <= 1000:
                # 60% modifier
                score = score + self.co2 * 1.6
        self.score = score
        return score

    def __str__(self):
        """Override of string representation

        :return: str, human-readable string output of the vehicle class
        """
        return 'id: {}, \nclass: {}, \nco2: {}, \nbb: {}, \nbb area: {}, \nstatus: {}, \nconfirmed: {}, \ndistances: {}, \nscore: {}' \
            .format(self.track_id, self.vehicle_class, self.co2, self.bb, self.bb_area, self.status, self.confirmed,
                    self.distances, self.score)


if __name__ == "__main__":
    test_vehicle = Vehicle(1, 2, [10, 10, 10, 10], 999)
    test_vehicle.distances = [14333, 10750, 14333, 14333, 14333, 11466, 12285, 14333, 14333, 12285, 10750, 12285, 12285, 12285, 8600, 10750, 10750,8600, 9555, 6142, 9555, 9555, 9555, 9555, 9555, 8600,7818, 7818, 8190, 7166, 7166, 7166, 6972, 6972, 7166, 4777, 6142, 5375, 5375, 4690, 4566, 4479, 4387, 4246, 4300, 4257, 4095, 3909, 3909, 3739, 3739, 3659]
    score = test_vehicle.get_score(25)
    print(score)
