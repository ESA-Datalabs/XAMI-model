import xml.etree.ElementTree as ET
import numpy as np
from skimage import measure
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from xml.dom import minidom

def binary_image_to_polygon(binary_image):
    array = binary_image.astype(int).reshape(binary_image.shape[0], binary_image.shape[1])
    # Find contours
    contours = measure.find_contours(array, 0.8)
    polygons = [contour.flatten().tolist() for contour in contours]
    return polygons

def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_list(item) for item in obj]
    else:
        return obj
        
def plot_polygon(polygon, image):
        xs = polygon[::2]  # Get every other element, starting from index 0
        ys = polygon[1::2] 
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='gray')

        # Create a polygon patch
        polygon = patches.Polygon(list(zip(ys, xs)), fill=False)
        ax.add_patch(polygon)
        ax.set_xlim(0, 255)
        ax.set_ylim(255, 0)
        
        plt.show()
    
def create_annotation(filename, width, height, depth, objects):
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, "folder").text = ''
    ET.SubElement(annotation, 'filename').text = filename
    ET.SubElement(annotation, 'path').text = filename
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'roboflow.com'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    ET.SubElement(annotation, 'segmented').text = '0'

    for obj in objects:
        object_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(object_elem, 'name').text = obj['name']
        ET.SubElement(object_elem, 'pose').text = 'Unspecified'
        ET.SubElement(object_elem, 'truncated').text = '0'
        ET.SubElement(object_elem, 'difficult').text = '0'
        ET.SubElement(object_elem, 'occluded').text = '0'

        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['bbox'][0])
        ET.SubElement(bndbox, 'ymin').text = str(obj['bbox'][1])
        ET.SubElement(bndbox, 'xmax').text = str(obj['bbox'][2]+obj['bbox'][0])
        ET.SubElement(bndbox, 'ymax').text = str(obj['bbox'][3]+obj['bbox'][1])
        
        polygon = ET.SubElement(object_elem, 'polygon')
        for i in range(0, len(obj['segmentations']), 2):

            ET.SubElement(polygon, f'x{i//2+1}').text = str(obj['segmentations'][i+1]+1.5)
            ET.SubElement(polygon, f'y{i//2+1}').text = str(obj['segmentations'][i]+1.5)  
    xml_str = ET.tostring(annotation, encoding='utf-8')

    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml()

    with open(f'{filename.replace(".png", ".xml")}', 'w') as f:
        f.write(pretty_xml)