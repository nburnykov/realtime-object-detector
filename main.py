from multiprocessing import Process
from detector.frames_capture import frames_capture
from detector.frames_detection import frames_detection
from detector.frames_augmentation import frames_augmentation
from detector.bbox_cut_out import bbox_cut_out

if __name__ == '__main__':
    capture = Process(target=frames_capture)
    detection = Process(target=frames_detection)
    augmentation = Process(target=frames_augmentation)
    bboxes = Process(target=bbox_cut_out)

    capture.start()
    detection.start()
    bboxes.start()

    # capture.join()
    # detection.join()
    # augmentation.join()
