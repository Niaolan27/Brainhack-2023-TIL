#this is where you place the cv 




class ObjectReIDService():
    '''
    Implementation of the Object Re-ID service.
    '''
    
    def __init__(self, yolo_model_path:str, reid_model_path:str, device=None):
        pass
        
    
    def targets_from_image(self, scene_img, target_img) -> BoundingBox:
        '''Process image with re-id pipeline and return the bounding box of the target_img.
        Returns None if the model doesn't believe that the target is within scene.

        Parameters
        ----------
        scene_img : ndarray
            Input image representing the scene to search through.

        target_img: ndarray
            Target image representing the object to re-identify.
        
        Returns
        -------
        results  : BoundingBox or None
            BoundingBox of target within scene.
            Assume the values are NOT normalized, i.e. the bbox values are based on the raw 
            pixel coordinates of the `scene_img`.
        '''
        yolo_model = YOLO(self.yolo_model_path)
        results = yolo_model(scene_img)
        objects_detected = []
        boxes = results[0].boxes   
	    for box in boxes:
            x,y,w,h = torch.squeeze(box.xywh).tolist()
            bbox = BoundingBox(x,y,w,h)
            plushie = img[int(y1):int(y2), int(x1):int(x2)]        
            classification, _ = infer(reid_model, plushie, target_img, Transforms())
            objects_detected.append({'class': classification.item(), 'bbox':bbox})
        for objects in objects_detected:
            if objects[class] == 1:
                return objects[bbox]
        return None
