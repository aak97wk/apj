
import jittor as jt
from jittor import init
from jittor import nn
'Bounding Box List definition.\n\nBoxList represents a list of bounding boxes as tensorflow\ntensors, where each bounding box is represented as a row of 4 numbers,\n[y_min, x_min, y_max, x_max].  It is assumed that all bounding boxes\nwithin a given list correspond to a single image.  See also\nbox_list_ops.py for common box related operations (such as area, iou, etc).\n\nOptionally, users can add additional related fields (such as weights).\nWe assume the following things to be true about fields:\n* they correspond to boxes in the box_list along the 0th dimension\n* they have inferable rank at graph construction time\n* all dimensions except for possibly the 0th can be inferred\n  (i.e., not None) at graph construction time.\n\nSome other notes:\n    * Following tensorflow conventions, we use height, width ordering,\n        and correspondingly, y,x (or ymin, xmin, ymax, xmax) ordering\n    * Tensors are always provided as (flat) [N, 4] tensors.\n'

class BoxList(object):
    'Box collection.'

    def __init__(self, boxes):
        'Constructs box collection.\n\n        Args:\n            boxes: a tensor of shape [N, 4] representing box corners\n\n        Raises:\n            ValueError: if invalid dimensions for bbox data or if bbox data is not in float32 format.\n        '
        if ((len(boxes.shape) != 2) or (boxes.shape[(- 1)] != 4)):
            raise ValueError('Invalid dimensions for box data.')
        if (boxes.dtype != jt.float32):
            raise ValueError('Invalid tensor type: should be tf.float32')
        self.data = {'boxes': boxes}

    def num_boxes(self):
        'Returns number of boxes held in collection.\n\n        Returns:\n          a tensor representing the number of boxes held in the collection.\n        '
        return self.data['boxes'].shape[0]

    def get_all_fields(self):
        'Returns all fields.'
        return self.data.keys()

    def get_extra_fields(self):
        "Returns all non-box fields (i.e., everything not named 'boxes')."
        return [k for k in self.data.keys() if (k != 'boxes')]

    def add_field(self, field, field_data):
        'Add field to box list.\n\n        This method can be used to add related box data such as weights/labels, etc.\n\n        Args:\n            field: a string key to access the data via `get`\n            field_data: a tensor containing the data to store in the BoxList\n        '
        self.data[field] = field_data

    def has_field(self, field):
        return (field in self.data)

    @property
    def boxes(self):
        'Convenience function for accessing box coordinates.\n\n        Returns:\n            a tensor with shape [N, 4] representing box coordinates.\n        '
        return self.get_field('boxes')

    @boxes.setter
    def boxes(self, boxes):
        'Convenience function for setting box coordinates.\n\n        Args:\n            boxes: a tensor of shape [N, 4] representing box corners\n\n        Raises:\n            ValueError: if invalid dimensions for bbox data\n        '
        if ((len(boxes.shape) != 2) or (boxes.shape[(- 1)] != 4)):
            raise ValueError('Invalid dimensions for box data.')
        self.data['boxes'] = boxes

    def get_field(self, field):
        'Accesses a box collection and associated fields.\n\n        This function returns specified field with object; if no field is specified,\n        it returns the box coordinates.\n\n        Args:\n            field: this optional string parameter can be used to specify a related field to be accessed.\n\n        Returns:\n            a tensor representing the box collection or an associated field.\n\n        Raises:\n            ValueError: if invalid field\n        '
        if (not self.has_field(field)):
            raise ValueError((('field ' + str(field)) + ' does not exist'))
        return self.data[field]

    def set_field(self, field, value):
        'Sets the value of a field.\n\n        Updates the field of a box_list with a given value.\n\n        Args:\n            field: (string) name of the field to set value.\n            value: the value to assign to the field.\n\n        Raises:\n            ValueError: if the box_list does not have specified field.\n        '
        if (not self.has_field(field)):
            raise ValueError(('field %s does not exist' % field))
        self.data[field] = value

    def get_center_coordinates_and_sizes(self):
        'Computes the center coordinates, height and width of the boxes.\n\n        Returns:\n            a list of 4 1-D tensors [ycenter, xcenter, height, width].\n        '
        box_corners = self.boxes
        (ymin, xmin, ymax, xmax) = box_corners.T.unbind()
        width = (xmax - xmin)
        height = (ymax - ymin)
        ycenter = (ymin + (height / 2.0))
        xcenter = (xmin + (width / 2.0))
        return [ycenter, xcenter, height, width]

    def transpose_coordinates(self):
        'Transpose the coordinate representation in a boxlist.\n\n        '
        (y_min, x_min, y_max, x_max) = self.boxes.chunk(4, dim=1)
        self.boxes = jt.contrib.concat([x_min, y_min, x_max, y_max], dim=1)

    def as_tensor_dict(self, fields=None):
        'Retrieves specified fields as a dictionary of tensors.\n\n        Args:\n            fields: (optional) list of fields to return in the dictionary.\n                If None (default), all fields are returned.\n\n        Returns:\n            tensor_dict: A dictionary of tensors specified by fields.\n\n        Raises:\n            ValueError: if specified field is not contained in boxlist.\n        '
        tensor_dict = {}
        if (fields is None):
            fields = self.get_all_fields()
        for field in fields:
            if (not self.has_field(field)):
                raise ValueError('boxlist must contain all specified fields')
            tensor_dict[field] = self.get_field(field)
        return tensor_dict

    @property
    def device(self):
        return self.data['boxes'].device
