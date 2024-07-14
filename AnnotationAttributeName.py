# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/07/15
# AnnotationAttributeName.py

import glob
import json
import traceback

class AnnotationAttributeName:

  def __init__(self):
    pass

  def listup(self, jsons_dir):
    json_files = glob.glob(jsons_dir + "/*.json")
    names = {""}
    for json_file in json_files:
        with open(json_file, "r") as f:
          data =json.load(f)
          annotations = data.get("Annotations")
          xannotations  = annotations.get("Annotation")
          for annotation in xannotations:
            if annotation == None or type(annotation) == str:
              continue
            attributes   = annotation.get("Attributes")
            attribute    = attributes.get("Attribute")
            name        = attribute.get("Name")
            names.add(name)
            print("--- name {}".format(name))

    print("=== names {}".format(names))


if __name__ == "__main__":
  try:
     name = AnnotationAttributeName()
     name.listup("./MoNuSAC-master-back/masks")

  except:
    traceback.print_exc()
