import xmltodict
import json
import os


class HTMLFormatter():

    # init method or constructor
    def __init__(self):
       
        self.cwd = os.getcwd()

    # Convert Html to Json Method
    def html_to_json_(self, file):
        pass


    # Convert xml to Json Method
    def xml_to_json_(self):
        file_dir = self.cwd + '/TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/code/global/htm'
        write_dir = self.cwd + '/TC-Financial-Report-Auto-Tagger-NLP-Ideation-Challenge/code/global/json/'
        for root,dirs,files in os.walk(file_dir, topdown=True):
            for file in files:
                with open(os.path.join(root, file)) as xml_file:
                    data_dict = xmltodict.parse(xml_file.read())
                    with open(write_dir + file + '.json', 'w') as writer:
                        writer.write( json.dumps(data_dict))
               
                return "Finished Writing..."
                            

                     


      
           
           
            

