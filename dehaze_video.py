"""Single image dehazing."""
from __future__ import division
import cv2
import numpy as np
import sys
import signal

class Channel_value:
    val = -1.0
    intensity = -1.0

def handler(signal, frame):
    global exit_signal
    exit_signal = True

def find_intensity_of_atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [Channel_value()] * top_num
    dark_channel = find_dark_channel(img)

    for y in xrange(img.shape[0]):
        for x in xrange(img.shape[1]):
            val = img.item(y, x, dark_channel)
            intensity = gray.item(y, x)
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break

    max_channel = Channel_value()
    for t in toplist:
        if t.intensity > max_channel.intensity:
            max_channel = t

    return max_channel.intensity


def find_dark_channel(img):
    return np.unravel_index(np.argmin(img), img.shape)[2]


def clamp(minimum, x, maximum):
    return max(minimum, min(x, maximum))


def dehaze(img, light_intensity, windowSize, t0, w):
    size = (img.shape[0], img.shape[1])

    outimg = np.zeros(img.shape, img.dtype)

    for y in xrange(size[0]):
        for x in xrange(size[1]):
            x_low = max(x-(windowSize//2), 0)
            y_low = max(y-(windowSize//2), 0)
            x_high = min(x+(windowSize//2), size[1])
            y_high = min(y+(windowSize//2), size[0])

            sliceimg = img[y_low:y_high, x_low:x_high]

            dark_channel = find_dark_channel(sliceimg)
            t = 1.0 - (w * img.item(y, x, dark_channel) / light_intensity)

            outimg.itemset((y,x,0), clamp(0, ((img.item(y,x,0) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,1), clamp(0, ((img.item(y,x,1) - light_intensity) / max(t, t0) + light_intensity), 255))
            outimg.itemset((y,x,2), clamp(0, ((img.item(y,x,2) - light_intensity) / max(t, t0) + light_intensity), 255))
    return outimg

def main():
    global exit_signal
    if not (len(sys.argv) == 3):
        print 'usage: python dehaze.py path_to_input_video.mp4 sampling_rate([frame]. when sampling_rate=1, all frames are usead.)'
    video_path = sys.argv[1]
    step = int(sys.argv[2])
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print 'cannot open video file'
        sys.exit(-1)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_path = video_path + 'out.mp4'
    writer = cv2.VideoWriter(out_path, fmt, int(video.get(cv2.CAP_PROP_FPS)), size)
    exit_signal = False
    signal.signal(signal.SIGINT, handler)
    count = 0
    frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
    while not exit_signal:
        for i in range(step):
            print str(count) + ' / ' + str(frame_num)
            count = count + 1
            ret, frame = video.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        light_intensity = find_intensity_of_atmospheric_light(frame, gray)
        w = 0.95
        t0 = 0.55
        outimg = dehaze(frame, light_intensity, 20, t0, w)
        writer.write(outimg)
    writer.release()

if __name__ == "__main__": main()
