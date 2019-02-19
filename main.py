import sys
import cv2
import numpy


def has_kill_info(frame):
    """
    :param frame: numpy ndarray of the frame CONTRACT: should be 1080p fullhd BGR
    :return: boolean wether the frame has kill info on it
    """
    assert frame is not None
    assert frame.shape == (
        1080, 1920, 3), f"Shape should be (1080, 1920, 3), got {frame.shape}"

    template = cv2.imread("kill.jpg")
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    roi = frame
    roi = frame[numpy.ix_([int(720 + i) for i in range(150)],
                          [int(780 + i) for i in range(350)])]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 17)),
    )

    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    threshold = 0.7
    loc = numpy.where(res >= threshold)
    return len(loc[0]) > 0


def main():
    """
    Script input: path to your gameplay video
    Outputs a "highlights.avi" file as result
    """

    video_path = sys.argv[1]

    video = cv2.VideoCapture(video_path)
    assert video.isOpened(), f"[{video_path}] could not be opened!"

    success, frame = video.read()
    kills = []
    while success:
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)
        timestamp = int(timestamp)

        # logic

        hki = has_kill_info(frame)
        if hki:
            kills.append(timestamp)

        for _ in range(20):
            success, frame = video.read()

    result = cv2.VideoWriter('highlight.avi',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15,
                             (1920, 1080))

    print("Kill times: ", kills)
    if not kills:
        return

    offset_ms = 3000
    last = 0
    video = cv2.VideoCapture(video_path)
    for kill in kills[1:]:
        if kill - last < 5000:
            continue
        last = kill
        kill -= offset_ms

        video.set(cv2.CAP_PROP_POS_MSEC, kill)

        success, frame = video.read()
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)

        while success and timestamp < kill + 6000:
            result.write(frame)
            success, frame = video.read()
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)


if __name__ == '__main__':
    main()
