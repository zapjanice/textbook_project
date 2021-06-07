import pandas as pd
import numpy as np
import cv2
import requests, io
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR, draw_ocr
from langdetect import detect
import enchant

from linalgo.annotate.models import Document, Annotation, Entity
from linalgo.hub.client import LinalgoClient
from linalgo.annotate.bbox import draw_bounding_boxes

class CreatingDataframe:

    def __init__(self, MYTOKEN):    
        client = LinalgoClient(token=MYTOKEN, api_url='https://prod.linhub.api.linalgo.com/v1')
        self.task = client.get_task('a9b4a03b-3af5-476f-8656-c69a32ea9866', verbose=True)
        
    def create_boundary_boxes(self, task, path):
        docs = []
        responses = []
        results = []
        for i in range(len(task.documents)):
            img_path = f'{path}/Img{i}.png'
            docs.append(task.documents[i])
            responses.append(requests.get(docs[i].content))
            Image.open(io.BytesIO(responses[i].content)).save(img_path)
            ocr = PaddleOCR(lang="japan")
            for i in range(len(docs)):
                results.append(ocr.ocr(img_path))
                for line in results[i]:
                    line.append(docs[i])
        return results
    
    def make_annotations(self, results):
        boxes = []
        texts = []
        scores = []
        doc_number = []
        for result in results:
            for line in result:
                boxes.append(line[0])
                texts.append(line[1][0])
                scores.append(line[1][1])
                doc_number.append(line[2])
        return boxes, texts, scores, doc_number
        
    def get_coordinates(self, boxes):
        top = []
        bottom = []
        left = []
        right = []
        for i in range(len(boxes)):
            bottom.append(boxes[i][2][1])
            top.append(boxes[i][0][1])
            left.append(boxes[i][0][0])
            right.append(boxes[i][1][0])
        return top, bottom, left, right
    
    def filter_numbers(self, texts):
        language = []
        for text in texts:
            try:
                language.append(detect(text))
            except: 
                language.append('number')
        return language
                
    def spellcheck(self, texts, language):
        d = enchant.Dict('en')
        for index, item in enumerate(texts):
            if language[index] not in ['ja', 'cn', 'zh-cn', 'zh-tw', 'number']:
                words = texts[index].split()
                words_trans = []
                for word in words:
                    if word.isalpha() == True:
                        if d.check(word) == False and d.suggest(word):
                            words_trans.append(d.suggest(word)[0])
                        else: 
                            words_trans.append(word)
                    else:
                        words_trans.append(word)
                separator = ' '
                texts[index] = separator.join(words_trans)
        return texts
        
    def make_dataframe(self, top, bottom, left, right, texts, language, doc_number):
        df = pd.DataFrame(list(top, bottom, left, right, texts, language, doc_number)), columns=['top','bottom', 'left', 'right', 'text', 'language', 'doc_number'])
        return df
    
    def make_item_bounding_boxes(self, df):
        df_item_boxes = df[df['text'].astype(str).str.isdigit()]
        df_item_boxes['bottom'] = df_item_boxes['top'].shift(-1)
        df_item_boxes = df_item_boxes.dropna()
        df_item_boxes.loc[df_item_boxes['language'] == 'number', 'language'] = 'item'
        return df_item_boxes
    
    def merge_dataframes(self, df, df_item_boxes):
        df_complete = pd.concat([df, df_item_boxes])
        return df_complete
    
    def run_function(self):
        results = self.create_boundary_boxes(self.task)
        boxes, texts, scores, doc_number = self.make_annotations(results)
        top, bottom, left, right = self.get_coordinates(boxes)
        language = self.filter_numbers(texts)
        df = self.make_dataframe(top, bottom, left, right, texts, language, doc_number)
        df_item_boxes = self.make_item_bounding_boxes(self, df)
        df_complete = self.merge_dataframes(df, df_item_boxes)
        return df_complete
