import haversine as hs
from haversine import Unit

from sklearn import preprocessing


def process_data(traj):
    distanceUnoccupied = 0
    distanceOccupied = 0
    prevLong = traj[0][0]
    prevLat = traj[0][1]
    feature = []
    for row in traj:
        try:
            if (row[-1] == 0):
                distanceUnoccupied += hs.haversine(
                    (prevLat, prevLong), (row[1], row[0]), unit=Unit.KILOMETERS)
            else:
                distanceOccupied += hs.haversine(
                    (prevLat, prevLong), (row[1], row[0]), unit=Unit.KILOMETERS)
        except Exception as e:
            print(e)
            print("Invalid data point", row)
            break
        prevLat = row[1]
        prevLong = row[0]
    feature.append([distanceOccupied, distanceUnoccupied])

    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(feature)


def run(data, model):
    predictions = model.predict(data)[0]
    plate = predictions.argmax()+1
    return plate
