import sys
import cv2
import numpy

TEMPLATE = cv2.imread("kill.jpg")
TEMPLATE = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)

TEMPLATE_MATCH_THRESHOLD = 0.7

HIGHLIGHT_OFFSET_MS = 3000
KILL_TEXT_ON_UI_DURATION_MS = 5000


def has_kill_info(frame):
    """
    :param frame: numpy ndarray of the frame CONTRACT: should be 1080p fullhd BGR
    :return: boolean wether the frame has kill info on it
    """
    assert frame is not None
    assert frame.shape == (
        1080, 1920, 3), f"Shape should be (1080, 1920, 3), got {frame.shape}"

    roi = frame[numpy.ix_([int(720 + i) for i in range(150)],
                          [int(780 + i) for i in range(350)])]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.morphologyEx(
        roi,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 17)),
    )

    res = cv2.matchTemplate(roi, TEMPLATE, cv2.TM_CCOEFF_NORMED)
    loc = numpy.where(res >= TEMPLATE_MATCH_THRESHOLD)
    return len(loc[0]) > 0


def main():
    """
    Script input: path to your gameplay video
    Outputs a "highlights.avi" file as result
    """

    video_path = sys.argv[1]

    video = cv2.VideoCapture(video_path)
    assert video.isOpened(), f"[{video_path}] could not be opened!"

    fps = video.get(cv2.CAP_PROP_FPS)
    kills = []

    success, frame = video.read()
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
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                             fps // 2, (1920, 1080))

    print("Kill times: ", kills)
    if not kills:
        return

    last = 0
    video = cv2.VideoCapture(video_path)
    for kill in kills:
        if kill - last < KILL_TEXT_ON_UI_DURATION_MS:
            continue
        last = kill

        kill = max(kill - HIGHLIGHT_OFFSET_MS, 0)

        video.set(cv2.CAP_PROP_POS_MSEC, kill)

        success, frame = video.read()
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC)

        while success and timestamp < kill + HIGHLIGHT_OFFSET_MS * 2:
            result.write(frame)
            success, frame = video.read()
            timestamp = video.get(cv2.CAP_PROP_POS_MSEC)


if __name__ == '__main__':
    main()
