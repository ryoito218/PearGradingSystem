import json
import triton_python_backend_utils as pb_utils
from mmdet.apis import init_detector, inference_detector
# from mmdet.registry import VISUALIZERS
from mmengine import Config
from mmengine.visualization import Visualizer
from mmengine.runner import Runner, set_random_seed
import numpy as np
import torch
import cv2
import time


def get_detection_model():
    model_root = '/models/pear_evaluator/1/maskrcnn'
    model_weight = f'{model_root}/model_weight.pth'
    cfg = Config.fromfile(f'{model_root}/config.py')
    cfg.metainfo = {
        'classes': ('alternaria', 'injury', 'speckle', 'plane', 'chemical', 'shape'),
        'palette': [
            (255, 0, 0),
            (0, 0, 255),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 255, 0),
        ]
    }

    cfg.data_root = '/models/pear_evaluator/1/maskrcnn'
    
    cfg.train_dataloader.dataset.ann_file = 'train.json'
    cfg.train_dataloader.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.metainfo = cfg.metainfo
    
    cfg.val_dataloader.dataset.ann_file = 'val.json'
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo

    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator.ann_file = cfg.data_root+'/'+'val.json'
    cfg.test_evaluator = cfg.val_evaluator

    cfg.work_dir = '/models/pear_evaluator/1/maskrcnn/work_dir'
    
    cfg.model.roi_head.bbox_head.num_classes = 6
    cfg.model.roi_head.mask_head.num_classes = 6

    set_random_seed(0, deterministic=False)

    print("######################################################")
    print(torch.cuda.is_available())
    print("######################################################")
    runner = Runner.from_cfg(cfg)
    model = init_detector(cfg, model_weight, device='cuda')
    return model


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.detection_model = get_detection_model()

        self.model_config = model_config = json.loads(args['model_config'])

        area_config = pb_utils.get_output_config_by_name(model_config, "AREA")
        number_config = pb_utils.get_output_config_by_name(model_config, "NUMBER")
        output_image_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT_IMAGE")
        speed_config = pb_utils.get_output_config_by_name(model_config, "SPEED")

        self.area_dtype = pb_utils.triton_string_to_numpy(area_config['data_type'])
        self.number_dtype = pb_utils.triton_string_to_numpy(number_config['data_type'])
        self.output_image_dtype = pb_utils.triton_string_to_numpy(output_image_config['data_type'])
        self.speed_dtype = pb_utils.triton_string_to_numpy(speed_config['data_type'])

        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate every request and create a
        # pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Perform inference on the request and append it to responses list...

            start_time = time.time()
            image = pb_utils.get_input_tensor_by_name(request, "IMAGE").as_numpy()
            end_time = time.time()
            load_image_time = end_time - start_time

            print(torch.cuda.is_available())

            start_time = time.time()
            detection_inference_results = inference_detector(self.detection_model, image)
            end_time = time.time()
            inference_time = end_time - start_time

            start_time = time.time()

            visualizer_now = Visualizer.get_current_instance()
            visualizer_now.dataset_meta = self.detection_model.dataset_meta
            visualizer_now.add_datasample(
                'detection_inference_result',
                image,
                data_sample=detection_inference_results,
                draw_gt=False,
                wait_time=0,
                out_file=f'/models/pear_evaluator/1/images/pear.png',
                pred_score_thr=0.3
            )
            
            end_time = time.time()
            visualize_time = end_time - start_time
            
            bbox_result = detection_inference_results.pred_instances.bboxes
            segm_result = detection_inference_results.pred_instances.masks
            class_result = detection_inference_results.pred_instances.labels
            score_result = detection_inference_results.pred_instances.scores

            # bbox_result_tensor = pb_utils.Tensor("BOUNDING_BOX_RESULT", bbox_result.astype(self.bbox_result_dtype))
            # segm_result_tensor = pb_utils.Tensor("SEGMENTATION_RESULT", segm_result.astype(self.segm_result_dtype))
            # class_result_tensor = pb_utils.Tensor("CLASS_RESULT", class_result.astype(self.class_result_dtype))

            start_time = time.time()
            index = torch.where(class_result == 5)[0]
            bbox = bbox_result[index][0]
            center_index = int((bbox[2]+bbox[0])//2)
            remaining_region = int(((bbox[2]-bbox[0])*np.sqrt(3))//4)
            end_time = time.time()
            bbox_time = end_time - start_time

            # print(center_index, remaining_region)

            start_time = time.time()
            new_masks = torch.tensor([]).to('cuda')
            for mask in segm_result:
                mask = mask[:,center_index-remaining_region:center_index+remaining_region]
                new_masks = torch.cat((new_masks, mask.unsqueeze(0)), dim=0).bool()
            end_time = time.time()
            remove_time = end_time - start_time
    

            area_class0 = area_class1 = area_class2 = area_class3 = area_class4 = area_class5 = 0
            num_class0 = 0

            start_time = time.time()
            for i in range(len(class_result)):
                label = class_result[i]
                probability = score_result[i]
                threshold = 0.3
                if label == 0 and torch.sum(new_masks[i]).item() and probability>=0.3:
                    area_class0 += torch.sum(new_masks[i]).item()
                    num_class0 += 1
                elif label == 1 and probability>=threshold:
                    area_class1 += torch.sum(new_masks[i]).item()
                elif label == 2 and probability>=threshold:
                    area_class2 += torch.sum(new_masks[i]).item()
                elif label == 3 and probability>=threshold:
                    area_class3 += torch.sum(new_masks[i]).item()
                elif label == 4 and probability>=threshold:
                    area_class4 += torch.sum(new_masks[i]).item()
                elif label == 5 and probability>=threshold:
                    area_class5 += torch.sum(new_masks[i]).item()
            end_time = time.time()
            area_time = end_time - start_time
    
            areas = [area_class0, area_class1, area_class2, area_class3, area_class4, area_class5]
            numpy_area = np.array(areas)
            numpy_number = np.array([num_class0])
            output_image_path = "/models/pear_evaluator/1/images/pear.png"

            start_time = time.time()
            output_image = cv2.imread(output_image_path)
            end_time = time.time()
            output_image_time = end_time - start_time
            
            numpy_output_image = np.array(output_image)
            speeds = [load_image_time, inference_time, visualize_time, bbox_time, remove_time, area_time, output_image_time]
            numpy_speeds = np.array(speeds)

            area_result_tensor = pb_utils.Tensor("AREA", numpy_area.astype(self.area_dtype))
            number_result_tensor = pb_utils.Tensor("NUMBER", numpy_number.astype(self.number_dtype))
            output_image_result_tensor = pb_utils.Tensor("OUTPUT_IMAGE", numpy_output_image.astype(self.output_image_dtype))
            speed_result_tensor = pb_utils.Tensor("SPEED", numpy_speeds.astype(self.speed_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[area_result_tensor, number_result_tensor, output_image_result_tensor, speed_result_tensor]
            )
            responses.append(inference_response)
        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')