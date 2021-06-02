!pip install git+git://github.com/linalgo/linalgo-sdk.git requests pillow
%load_ext autoreload
%autoreload 2
import requests, io
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from linalgo.annotate.models import Document, Annotation, Entity
from linalgo.hub.client import LinalgoClient
from linalgo.annotate.bbox import draw_bounding_boxes

class Making_DF:

    def __init__(self):    
        client = LinalgoClient(token=MYTOKEN, api_url='https://prod.linhub.api.linalgo.com/v1')
        self.task = client.get_task('a9b4a03b-3af5-476f-8656-c69a32ea9866', verbose=True)

    def making_list(task): 
        ent_list =[]
        coord_list = []
        doc_list = []
        for i in range(len(task.documents)): 
            document = task.documents[i]
            for ant in range(len(document.annotations)): 
                annotation = document.annotations[ant]
                entity = annotation.entity.name
                coordinates = annotation.target.selectors[0]
                ent_list.append(entity)
                coord_list.append(coordinates)
                doc_list.append(document.content)
            return  ent_list, coord_list, doc_list

    def getting_coordinates(coord_list): 
        top= []
        bottom=[]
        left=[]
        right=[]
        area = []
        width=[]
        height=[]
        for i in range(len(coord_list)):
            top.append(coord_list[i].top)
            bottom.append(coord_list[i].bottom)
            left.append(coord_list[i].left)
            right.append(coord_list[i].right)
            area.append(coord_list[i].area)
            width.append(coord_list[i].width)
            height.append(coord_list[i].height)
        return top, bottom, left, right, area, width, height

    def making_df(ent_list, coord_list, doc_list, top, bottom, left, right, area, width, height): 
        elem = {'Entity': ent_list, 'document_id': doc_list}
        df= pd.DataFrame(elem)
        df.columns = ['Entity', 'document_id']

        df['document_no'] = df.groupby(['document_id']).ngroup()

        df['top'] = pd.DataFrame(top)
        df['bottom'] = pd.DataFrame(bottom)
        df['left'] = pd.DataFrame(left)
        df['right'] = pd.DataFrame(right)
        df['area'] = pd.DataFrame(area)
        df['width'] = pd.DataFrame(width)
        df['height'] = pd.DataFrame(height)

        df = df.sort_values(by=['document_no', 'top'])
        df['entity_count'] = df.groupby(['document_id', 'Entity']).cumcount()
        df = df.reset_index(drop=True)
        df = df.dropna()
        return df

    def clean_data(df): 
        keep_df = df[df['Entity'] == 'word-en']
        keep_s = keep_df.document_no.reset_index(drop=True).tolist()
        df = df[df['document_no'].isin(keep_s)]
        return df
    
    def run_function(self):
        ent_list, coord_list, doc_list = making_list(self.task)
        top, bottom, left, right, area, width, height = getting_coordinates(coord_list)
        df = making_df(ent_list, coord_list, doc_list, top, bottom, left, right, area, width, height)
        return clean_data(df)
