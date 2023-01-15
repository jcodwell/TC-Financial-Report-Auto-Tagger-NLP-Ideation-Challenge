

import os

import pandas as pd
class JSONFormatter():

    # init method or constructor
    def __init__(self):
       
        self.cwd = os.getcwd()



    # Convert Json to TXT Method
    def json_txt(self):
        file_dir = self.cwd + '/TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/code/global/json/'
        write_dir = self.cwd + '/TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/code/global/text/'
        for root,dirs,files in os.walk(file_dir, topdown=True):
            for file in files:
                df = pd.read_json (os.path.join(root, file))
                df.to_csv ( write_dir + file + '.txt', index = False)

               #  print(file)
               #  with open(os.path.join(root, file)) as json_file:
               #       jsonObject = json.loads(json_file)

                  #   with open(write_dir + file + '.json', 'w') as writer:
                  #       writer.write( json.dumps(data_dict))
               
            return "Finished Writing..."
                            

HF =  JSONFormatter()
HF.json_txt()