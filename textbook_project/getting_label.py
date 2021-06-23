import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from linalgo.annotate.models import Document, Annotation, Entity
from linalgo.hub.client import LinalgoClient
from linalgo.annotate.bbox import draw_bounding_boxes
from tqdm import tqdm
import linalgo
from textbook_project.clean_data import Making_DF

'''Code use to match dataframe obtain from OCR tool 
with annotated data to get labels to make label classification model'''

class Getting_Label:

    def __init__(self, MYTOKEN):
        self.total_df = Making_DF(MYTOKEN).run_function()
        self.pred_df = pd.read_csv('all_documents_dataframe.csv')

    def add_features(self, pred_df):
        pred_df = pred_df.rename(columns={'doc_number':'document_id'})
        pred_df = pred_df.iloc[1:,:]
        pred_df['height']= pd.Series.abs(pred_df['bottom'] - pred_df['top'])
        pred_df['width']= pd.Series.abs(pred_df['right'] - pred_df['left'])
        pred_df['area'] = pd.Series(pred_df['height'] * pred_df['width'])
        pred_df['aspect_ratio'] = pd.Series(pred_df['height'] / pred_df['width'])
        return pred_df

    def get_item_df(self, pred_df):
        item_df = pred_df[pred_df['language'] == 'item']
        def label(x):
            for item in range(len(item_df.bottom.values)):
                if x <item_df.bottom.values[item]:
                    return item
                item=+item
            return item
        pred_df['item_no']  = pred_df['top'].apply(label)
        item_df['item_no'] = item_df.index
        item_df.rename(columns = {'top':'top_item', 'bottom':'bottom_item', 'left':'left_item',
                                  'right':'right_item'}, inplace = True)
        return item_df

    def merge_for_xy_diff(self, pred_df, item_df):
        pred_df = pred_df.merge(item_df[['top_item', 'left_item', 'item_no']], how='left', on=(['item_no']))
        pred_df['y_diff'] = pd.Series(pred_df['top_item'] - pred_df['top'])
        pred_df['x_diff'] = pd.Series(pred_df['left_item'] - pred_df['left'])
        pred_df['x_diff'].fillna(0, inplace = True)
        pred_df['y_diff'].fillna(0, inplace = True)
        pred_df = pred_df.drop(['top_item', 'left_item'], axis = 1)
        return pred_df

    def match_bounding_box(self, pred_df, ground_truth_df):
        data_set_df = pd.DataFrame(columns=pred_df.columns)
        with tqdm(total=pred_df.shape[0]) as pbar:
            for index, row in pred_df.iterrows():
                pred_bb = linalgo.annotate.bbox.BoundingBox(row.left, row.right, row.top, row.bottom)
                temp_lin = ground_truth_df[ground_truth_df.document_id == row.document_id]
                for index_lin, row_lin in temp_lin.iterrows():
                    lin_bb = linalgo.annotate.bbox.BoundingBox(row_lin.left, row_lin.right, row_lin.top, row_lin.bottom)
                    if pred_bb.overlap(lin_bb) > 0.51:
                        pred_df.loc[index, 'Entity'] = row_lin.Entity
                        data_set_df = data_set_df.append(pred_df.iloc[index], ignore_index=True)
                pbar.update(1)
        return data_set_df

    def run_function(self):
        pred_df = self.add_features(self.pred_df)
        item_df = self.get_item_df(pred_df)
        pred_df = self.merge_for_xy_diff(pred_df, item_df)
        pred_df = self.match_bounding_box(pred_df, self.total_df)
        train_no = int(len(pred_df)*0.8)
        pred_train_df = pred_df[:train_no]
        pred_test_df = pred_df[train_no:]
        return  pred_train_df, pred_test_df
