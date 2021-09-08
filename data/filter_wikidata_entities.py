from qwikidata.json_dump import WikidataJsonDump
from qwikidata.entity import WikidataItem, WikidataProperty, LabelDescriptionAliasMixin
import json
from tqdm import tqdm


filename = "/Volumes/LaCie/latest-all.json.bz2"

wjd = WikidataJsonDump(filename)

type_to_entity_class = {"item": WikidataItem, "property": WikidataProperty}
entities = []

keyword_ids = ['Q639669', 'Q638', 'Q177220', 'Q753110', 'Q488205', 'Q3455803', 'Q183945', 'Q2252262','Q486748','Q855091','Q36834','Q12800682']
keywords = ['musician', 'music', 'singer', 'songwriter', 'singer-songwriter', 'director', 'record producer', 'rapper','pianist','guitarist','composer','saxophonist']


for ii, entity_dict in tqdm(enumerate(wjd)):
    entity_id = entity_dict["id"]
    entity_type = entity_dict["type"]
    entity = type_to_entity_class[entity_type](entity_dict)

    for keyword in keywords:
        if keyword in entity.get_description():
            entities.append(entity_dict)
            break
        else:
            try:
                occupation = entity_dict['claims']['P106']
        
                for instance in occupation:
                    instance_id = instance['mainsnak']['datavalue']['value']['id']
                    if instance_id in keywords:
                        entities.append(entity_dict)
            except:
                continue
            

with open('/Volumes/LaCie/entities_21_07_21.json', 'w') as file:
    json.dump(entities, file)