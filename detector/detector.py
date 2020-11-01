import tensorflow as tf
import io
import numpy as np
from PIL import Image
import tensorflow.keras.backend as K

# TODO calculate anchor boxes using hardhat dataset, compare mAP with default COCO pretrained anchor boxes
ANCHOR_BOXES = ((10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (59, 119),
                (116, 90), (156, 198), (373, 326))

ANCHOR_MASK = ((6, 7, 8), (3, 4, 5),
               (0, 1, 2))  # 3 output layers x 3 anchors on every layer, first layer - small boxes, last - large boxes

BOX_THRESHOLD = 0.6
IOU_THRESHOLD = 0.5
MAX_BOXES_IN_LAYER = 40

ORIGINAL_OUTPUT_DEPTH = 255  # 3 anchor boxes x (4 box offsets + 1 objectiveness score + 80 class scores)


class YOLOv3:
    def __init__(self,
                 model_file: str,
                 classes: dict,
                 batch_size: int,
                 anchor_boxes: tuple = ANCHOR_BOXES,
                 anchor_mask: tuple = ANCHOR_MASK,
                 score_threshold: float = 0.6,
                 iou_threshold: float = 0.5,
                 boxes_num: int = MAX_BOXES_IN_LAYER):

        self.tf_graph = tf.Graph()
        self.tf_session = tf.Session(graph=self.tf_graph)

        self.model_file = model_file
        self.classes = classes
        self.n_classes = len(self.classes)
        self.batch_size = batch_size

        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.boxes_num = boxes_num

        with self.tf_session.graph.as_default():
            K.set_session(self.tf_session)
            self.model = tf.keras.models.load_model(self.model_file)

            self.n_anchors = self.model.output_shape[0][-1] // (len(classes) + 5)
            self.input_image_dims = self.model.input.shape[1:3]
            self.n_outputs = len(self.model.outputs)

            self.anchor_boxes = anchor_boxes
            self.anchor_mask = anchor_mask
            self.detection_pipeline = self._detection_graph_init()

        assert self.model.layers[-1].filters % (len(classes) + 5) == 0, \
            f'Internal model classes count and classes count ({len(classes)}) are incompatible'

        assert len(anchor_boxes) % self.n_anchors == 0, \
            f'Number of anchor boxes ({len(anchor_boxes)}) and model anchors count ({self.n_anchors}) are incompatible'

        assert len(anchor_mask) % self.n_anchors == 0, \
            f'Number of anchor mask sets ({len(anchor_mask)}) and model anchors count ({self.n_anchors}) are incompatible'

    def preprocess_image(self, image_bytes):
        """
        :param image_bytes: image as bytes string
        :return: boxes with classes after NMS
        uses PIL instead of tensorflow.image because of
        https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
        """
        bytes_io = io.BytesIO(image_bytes)
        image_pil = Image.open(bytes_io)

        if image_pil.size != self.input_image_dims:
            image_pil = image_pil.resize(self.input_image_dims)

        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        image_data = np.array(image_pil, dtype='float32')

        return self.preprocess_array(image_data)

    def preprocess_array(self, image_array):
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        image_array = image_array / 255

        return image_array

    def _split_output_tensor(self, tensor_output, n_anchors, n_classes):
        batch, width, height, depth = tensor_output.shape
        tensor_reshaped = K.reshape(tensor_output, [-1, width, height, n_anchors, n_classes + 5])
        xy = tensor_reshaped[..., :2]
        wh = tensor_reshaped[..., 2:4]
        objectiveness = tensor_reshaped[..., 4]
        classes = tensor_reshaped[..., 5:]
        return xy, wh, objectiveness, classes

    def _calculate_features(self, xy, wh, objectiveness, classes, anchors):
        shape = K.shape(xy)[1:3]  # width, height

        xy_sig = K.sigmoid(xy)
        # TODO rethink logic here, grid needs to be calculated just once after model initialization
        col = K.reshape(K.tile(K.arange(0, shape[0]), shape[0:1]), (-1, shape[0]))
        row = K.reshape(K.tile(K.arange(0, shape[1]), shape[1:2]), (-1, shape[1]))
        row = K.transpose(row)
        col = K.repeat_elements(K.reshape(col, (shape[0], shape[1], 1, 1)), rep=len(anchors), axis=-2)
        row = K.repeat_elements(K.reshape(row, (shape[0], shape[1], 1, 1)), rep=len(anchors), axis=-2)
        grid = K.concatenate((col, row), axis=-1)
        # TODO same thing for the anchors
        anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, len(anchors), 2])

        box_xy = (xy_sig + K.cast(grid, K.dtype(xy_sig))) / (shape[0], shape[1])

        box_wh = K.exp(wh) * anchors_tensor / K.cast(self.input_image_dims, K.dtype(wh))

        obj_sig = K.sigmoid(objectiveness)
        class_sig = K.sigmoid(classes)

        return box_xy, box_wh, obj_sig, class_sig

    def _detection_graph_init(self):
        nmsed_boxes = []
        nmsed_classes = []
        nmsed_scores = []
        for mask, output_layer in zip(self.anchor_mask, self.model.outputs):  # iterate over YOLO layers
            xy, wh, objectiveness, classes = self._split_output_tensor(output_layer, self.n_anchors,
                                                                       self.n_classes)
            anchors = [self.anchor_boxes[i] for i in mask]
            box_xy, box_wh, obj_sigmoid, class_sigmoid = self._calculate_features(xy, wh, objectiveness, classes,
                                                                                  anchors)
            sigmoid_shape = K.shape(obj_sigmoid)
            obj_sigmoid = K.reshape(obj_sigmoid,
                                    (sigmoid_shape[0], sigmoid_shape[1], sigmoid_shape[2], sigmoid_shape[3], 1))
            scores = class_sigmoid * obj_sigmoid

            x1y1 = box_xy - box_wh / 2
            x2y2 = x1y1 + box_wh
            boxes = K.concatenate((x1y1, x2y2), axis=-1)
            boxes_shape = K.shape(boxes)  # batch, grid_x, grid_y, anchors, rectangles(4)
            boxes = K.reshape(boxes,
                              (boxes_shape[0], boxes_shape[1] * boxes_shape[2] * boxes_shape[3], 1, boxes_shape[4]))
            boxes = K.cast(boxes, dtype=tf.float32)

            scores_shape = K.shape(scores)  # batch, grid_x, grid_y, anchors, scores
            scores = K.reshape(scores,
                               (scores_shape[0], scores_shape[1] * scores_shape[2] * scores_shape[3], scores_shape[4]))

            nms = tf.image.combined_non_max_suppression(boxes, scores, self.boxes_num, self.boxes_num,
                                                        score_threshold=self.score_threshold)  # NMS inside YOLO layer
            nmsed_boxes.append(nms.nmsed_boxes)
            nmsed_classes.append(nms.nmsed_classes)
            nmsed_scores.append(nms.nmsed_scores)

        nmsed_boxes = tf.unstack(K.concatenate(nmsed_boxes, axis=1),
                                 num=self.batch_size)  # concatenate over YOLO layers and unstack batch
        nmsed_classes = tf.unstack(K.concatenate(nmsed_classes), num=self.batch_size)
        nmsed_scores = tf.unstack(K.concatenate(nmsed_scores), num=self.batch_size)

        boxes_dict = {}
        for i, data in enumerate(zip(nmsed_boxes, nmsed_classes, nmsed_scores)):  # iterate over batch
            bxs, cls, scs = data
            mask = scs >= self.score_threshold

            bxs = tf.boolean_mask(bxs, mask)
            cls = tf.boolean_mask(cls, mask)
            scs = tf.boolean_mask(scs, mask)

            box_indexes = tf.image.non_max_suppression(bxs, scs, K.shape(scs)[0])
            boxes_dict[i] = dict(boxes=K.gather(bxs, box_indexes),
                                 classes=K.gather(cls, box_indexes),
                                 scores=K.gather(scs, box_indexes))

        return boxes_dict

    def single_detection(self, preprocessed_array: str):
        with self.tf_session.graph.as_default():
            boxes = self.tf_session.run(self.detection_pipeline, feed_dict={self.model.input: preprocessed_array})
        return boxes

    def batch_detection(self):
        pass
