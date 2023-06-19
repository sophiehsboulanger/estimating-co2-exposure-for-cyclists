class Vehicle:
    def __init__(self, track_id, vehicle_class, bb):
        self.track_id = track_id
        self.vehicle_class = vehicle_class
        self.bb = bb
        self.confirmed = False
        self.distances = []

