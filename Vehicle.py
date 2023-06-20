def get_bb_area(bb):
    x1 = bb[0]
    y1 = bb[1]
    x2 = bb[2]
    y2 = bb[3]

    width = x2 - x1
    height = y2 - y1

    return abs(width * height)


class Vehicle:
    def __init__(self, track_id, vehicle_class, bb, status):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.bb = bb
        self.bb_area = get_bb_area(bb)
        self.status = status
        self.confirmed = False
        self.distances = []

    def set_bb(self, bb):
        self.bb = bb
        self.bb_area = get_bb_area(bb)

    def __str__(self):
        return 'id: {}, \nclass: {}, \nbb: {}, \nbb area: {}, \nstatus: {}, \nconfirmed: {}, \ndistances: {}'\
            .format(self.track_id, self.vehicle_class, self.bb, self.bb_area, self.status, self.confirmed, self.distances)