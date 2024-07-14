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
