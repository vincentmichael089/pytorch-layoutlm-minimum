# %%
import pandas as pd
import os
import json
import fitz
import pandas as pd
os.environ['PATH'] += os.pathsep + "dir/to/textlive"+"/texdist/texlive/2022/bin/x86_64-linux"

def extractPDF(file_list, folder_path, output_path):
  # - file list is a list containing pdf's file name
  # - folder path is where the pdfs reside
  # - output path is where to generate the bbox (json) and the characters (pandas-csv)
  dictPyMu = {}

  for _, v in enumerate(file_list):
    target = v.replace(".png","")
    path = folder_path+target+".pdf"

    #pymu
    doc = fitz.open(path)
    parsed_pymu = doc.load_page(0).get_text()
    dictPyMu.update({
      v: {
        "raw_pymu":parsed_pymu,
        "clean_pymu":parsed_pymu.replace("\n","").replace(" ","")[:-3]
      }
    })

    #position json
    positionList = []
    for block in doc.load_page(0).get_text("rawdict")['blocks']:
      for line in block['lines']:
        for span in line["spans"]:
          for char in span["chars"]:
            positionList.append(char)

    dictPosition = {
      target : positionList
    }

    json_object = json.dumps(dictPosition, indent=4)

    with open(output_path+target+".json", "w") as outfile:
      outfile.write(json_object)

  df_pymu = pd.DataFrame.from_dict(dictPyMu,  orient='index')
  df_pymu.to_csv(output_path, index=False)


def bbox_json_to_sequence(x):
  # to be used by transformers, the bboxes needs to be transformed into a sequence like format.
  # for example, a sequence with a length of N will have bbox with a size of (N x 4).
  # sample usage (feel free to modify): df["raw_pymu_from_json"] = df["path"].apply(extract_json)
  try:
    f = open("/path/to/folder/containing/json/files"+x+".json")
    data = json.load(f)
    return "".join([i['c'] if i['c'] != " " else "<space>" for i in data[x]])
  except:
    pass


# from https://www.kaggle.com/code/sandeepkumarkushwaha/finetuning-layoutlm-v2-for-receipt-recognition-nlp/notebook
def normalize_bbox(bbox, width=595, height=842):
  # Normalize and discretize the bbox value to range 0-1000. 1000 is the value determined by the LayoutLM
  # later on any special tokens will have bbox value of [0,0,0,0]
  # width and height set to A4 paper
  return [
      int(1000 * (bbox[0] / width)),
      int(1000 * (bbox[1] / height)),
      int(1000 * (bbox[2] / width)),
      int(1000 * (bbox[3] / height)),
  ]