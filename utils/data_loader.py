import os
import re
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

class XMLDataLoader:
    def __init__(self, 
                 path_to_folder: str, 
                 is_convert_to_numbers=True,
                 is_split_text=True,
                 is_remove_excessive_new_lines=True):
        self.path_to_folder                 = path_to_folder
        self.is_convert_to_numbers          = is_convert_to_numbers
        self.is_split_text                  = is_split_text
        self.is_remove_excessive_new_lines  = is_remove_excessive_new_lines

        self.criteria = [
            'ABDOMINAL',
            'ADVANCED-CAD',
            'ALCOHOL-ABUSE',
            'ASP-FOR-MI',
            'CREATININE',
            'DIETSUPP-2MOS',
            'DRUG-ABUSE',
            'ENGLISH',
            'HBA1C',
            'KETO-1YR',
            'MAJOR-DIABETES',
            'MAKES-DECISIONS',
            'MI-6MOS',
        ]
        self.original_definitions = {
            'ABDOMINAL': 'History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction',
            'ADVANCED-CAD': 'Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: • Taking 2 or more medications to treat CAD • History of myocardial infarction (MI) • Currently experiencing angina • Ischemia, past or present',
            'ALCOHOL-ABUSE': 'Current alcohol use over weekly recommended limits',
            'ASP-FOR-MI': 'Use of aspirin for preventing myocardial infarction (MI)',
            'CREATININE': 'Serum creatinine level above the upper normal limit',
            'DIETSUPP-2MOS': 'Taken a dietary supplement (excluding vitamin D) in the past 2 months',
            'DRUG-ABUSE': 'Current or past history of drug abuse',
            'ENGLISH': 'Patient must speak English',
            'HBA1C': 'Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%',
            'KETO-1YR': 'Diagnosis of ketoacidosis within the past year',
            'MAJOR-DIABETES': 'Major diabetes-related complication. For the purposes of this annotation, we define “major complication” (as opposed to “minor complication”) as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: • Amputation • Kidney damage • Skin conditions • Retinopathy • nephropathy • neuropathy',
            'MAKES-DECISIONS': 'Patient must make their own medical decisions',
            'MI-6MOS': 'Myocardial infarction (MI) within the past 6 months'
        }
        # Custom definitions for better prompts
        self.definitions = {
            'ABDOMINAL': 'History of intra-abdominal surgery. This could include any form of intra-abdominal surgery, including but not limited to small/large intestine resection or small bowel obstruction',
            'ADVANCED-CAD': 'Advanced cardiovascular disease (CAD). For the purposes of this annotation, we define “advanced” as having 2 or more of the following: (a) Taking 2 or more medications to treat CAD (b) History of myocardial infarction (MI) (c) Currently experiencing angina (d) Ischemia, past or present. The patient must have at least 2 of these categories (a,b,c,d) to meet this criterion, otherwise the patient does not meet this criterion. For ADVANCED-CAD, be strict in your evaluation of the patient -- if they just have cardiovascular disease, then they do not meet this criterion.',
            'ALCOHOL-ABUSE': 'Current alcohol use over weekly recommended limits',
            'ASP-FOR-MI': 'Use of aspirin for preventing myocardial infarction (MI)..',
            'CREATININE': 'Serum creatinine level above the upper normal limit',
            'DIETSUPP-2MOS': "Consumption of a dietary supplement (excluding vitamin D) in the past 2 months. To assess this criterion, go through the list of medications_and_supplements taken from the note. If a substance could potentially be used as a dietary supplement (i.e. it is commonly used as a dietary supplement, even if it is not explicitly stated as being used as a dietary supplement), then the patient meets this criterion. Be lenient and broad in what is considered a dietary supplement. For example, a 'multivitamin' and 'calcium carbonate' should always be considered a dietary supplement if they are included in this list.",
            'DRUG-ABUSE': 'Current or past history of drug abuse',
            'ENGLISH': 'Patient speaks English. Assume that the patient speaks English, unless otherwise explicitly noted. If the patient\'s language is not mentioned in the note, then assume they speak English and thus meet this criteria.',
            'HBA1C': 'Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%',
            'KETO-1YR': 'Diagnosis of ketoacidosis within the past year',
            'MAJOR-DIABETES': 'Major diabetes-related complication. Examples of “major complication” (as opposed to “minor complication”) include, but are not limited to, any of the following that are a result of (or strongly correlated with) uncontrolled diabetes: • Amputation • Kidney damage • Skin conditions • Retinopathy • nephropathy • neuropathy. Additionally, if multiple conditions together imply a severe case of diabetes, then count that as a major complication.',
            'MAKES-DECISIONS': 'Patient must make their own medical decisions. Assume that the patient makes their own medical decisions, unless otherwise explicitly noted. There is no information provided about the patient\'s ability to make their own medical decisions, then assume they do make their own decisions and therefore meet this criteria."',
            'MI-6MOS': 'Myocardial infarction (MI) within the past 6 months'
        }
        # Decide how to "pool" preds per note for a single label for the overall patient
        self.criteria_2_agg: Dict[str, str] = {
            # Criteria that do not depend on the current date where 1 overrides everything will be `max`
            'ABDOMINAL' : 'max',
            'ADVANCED-CAD' : 'max',
            'ASP-FOR-MI' : 'max',
            'CREATININE' : 'max',
            'DRUG-ABUSE' : 'max',
            'ALCOHOL-ABUSE' : 'max',
            'HBA1C' : 'max',
            'MAJOR-DIABETES' : 'max',
            'MAKES-DECISIONS' : 'max',
            # Criteria that do not depend on the current date where 0 overrides everything will be `min`
            'ENGLISH' : 'min',
            # Criteria that do depend on the current date will be 'most_recent'
            'DIETSUPP-2MOS' : 'most_recent|2',
            'KETO-1YR' : 'most_recent|12',
            'MI-6MOS' : 'most_recent|6',
        }
     
    def get_definitions_as_string(self):
        dictionary_string = '\n'.join([f'-> {key}: {value}' for key, value in self.definitions.items()])
        return dictionary_string
    
    def get_definitions_as_list(self) -> List[str]:
        list_ = [f'{key}: {value}' for key, value in self.definitions.items()]
        return list_

    def load_data(self) -> List[Dict[str, Any]]:
        """ Main function: Data loader for the XML files"""
        data        = []
        file_names  = os.listdir(self.path_to_folder)
        file_names  = sorted([file for file in file_names  if  file.endswith('.xml')])
        for file_name in file_names:
            file_path = os.path.join(self.path_to_folder, file_name)
            text, labels = self.parse_xml(file_path)
            data.append({
                "patient_id": file_name.replace(".xml",""),
                "ehr": text,
                "labels": labels
            })

        return data

    @staticmethod
    def get_date_of_note(patient: Dict[str, Any], note_idx: int) -> str:
        """Get date of note for patient"""
        if not isinstance(note_idx, int):
            if isinstance(eval(note_idx), list):
                note_idx = sorted(eval(note_idx))[-1]
        assert note_idx <= len(patient['ehr']), f"{note_idx} out of bounds for {patient['patient_id']}"
        note: str = patient['ehr'][note_idx]
        match = re.search(r"Record date: (\d{4}-\d{2}-\d{2})", note)
        date = match.group(1) if match else None
        if not date:
            print(f"ERROR - Could not find the date for patient {patient['patient_id']}")
        return date
        
    @staticmethod
    def get_current_date_for_patient(patient: Dict[str, Any]) -> str:
        """Get most recent date visible in files for a given patient"""
        most_recent_date = None
        for note in patient['ehr']:
            match = re.search(r"Record date: (\d{4}-\d{2}-\d{2})", note)
            most_recent_date = match.group(1) if match else most_recent_date
        if not most_recent_date:
            print(f"ERROR - Could not find the date for patient {patient['patient_id']}")
        return most_recent_date
            

    def parse_xml(self, XML_file) -> Tuple[str, Dict[str, str]]:
        tree = ET.parse(XML_file) # Get the root element
        root = tree.getroot()     # Get the root element

        # Iterate over the elements and separate <TEXT> from <TAG>
        for elem in root.iter():
            if elem.tag == 'TEXT':
                text = elem.text
                if self.is_remove_excessive_new_lines:
                    text = self.remove_excessive_newlines(text)
                if self.is_split_text:
                    text = self.split_text(text)
            elif elem.tag == 'TAGS':
                tags = self.read_tags(root)
                
        return (text, tags)



    def read_tags(self, root) -> Dict[str, str]:
        """Reads the tags from an XML file and returns a dictionary of tags"""
        tags_dict = {}                                              # Initialize an empty dictionary
        for tag in root.iter('TAGS'):                               # Iterate over the subtags of <TAGS> and extract the met value
            for subtag in tag:                                      # Iterate over the subtags of <TAGS> and extract the met value
                met_value = subtag.attrib.get('met')                # Get the met value
                if self.is_convert_to_numbers:                              # Convert the met value to a number
                    met_value = 1 if met_value == 'met' else 0      # Convert the met value to a number
                tags_dict[subtag.tag] = met_value                   # Add the tag to the dictionary

        return tags_dict


    def split_text(self, text: str) -> List[str]:
        split_char = '*' * 100
        parts = [ x.strip() for x in text.split(split_char) if x.strip() != '' ]
        return parts

    def remove_excessive_newlines(self, text: str) -> str:
        text = text.replace('\n\n\n', '\n')
        return text