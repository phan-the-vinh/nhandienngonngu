import math


def normalize_landmarks(hand_landmarks):
    """
    Chuan hoa 21 landmark cua ban tay
    thanh 63 dac trung cho MLPClassifier.
    """

    # Landmark 0 la co tay
    wrist = hand_landmarks[0]

    # Luu cac toa do tuong doi so voi co tay
    relative_points = []

    for landmark in hand_landmarks:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        relative_points.append((x, y, z))

    # Tim khoang cach lon nhat tu co tay
    # de dung lam ti le chuan hoa kich thuoc ban tay
    max_distance = 0.0

    for x, y, z in relative_points:
        distance = math.sqrt(
            x * x +
            y * y +
            z * z
        )

        if distance > max_distance:
            max_distance = distance

    # Tranh chia cho 0
    if max_distance == 0:
        max_distance = 1.0

    # Tao danh sach 63 dac trung
    features = []

    for x, y, z in relative_points:
        x = x / max_distance
        y = y / max_distance
        z = z / max_distance

        features.extend([x, y, z])

    return features
