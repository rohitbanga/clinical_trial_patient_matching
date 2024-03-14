from collections import namedtuple
import pickle
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
from utils.data_loader import XMLDataLoader
from utils.helpers import batch_query_openai, batch_query_hf, model_2_tokens
from utils.types import CriterionAssessment, UsageStat
from typing import Dict, List, Any, Set, Tuple, Union

def truncate_doc(tokenizer, doc: str, max_tokens: int) -> str:
    """Truncate the document to a maximum number of tokens."""
    return tokenizer.decode(tokenizer.encode(doc)[:max_tokens])

def generate_result(patient: Dict[str, Any], note: str, note_idx: Union[int, List[int]], prompt: str, assessment: CriterionAssessment, stat: UsageStat) -> Dict[str, Any]:
    """Helper method that constructs a result dictionary from a patient, note, prompt, and assessment."""
    criterion: str = assessment.criterion
    rationale: str = assessment.rationale
    is_met: str = assessment.is_met
    confidence: str = assessment.confidence
    medications_and_supplements: List[str] = assessment.medications_and_supplements
    return {
        'patient_id' : patient['patient_id'],
        'note' :       note,
        'note_idx' :   note_idx,
        'criterion' :  criterion,
        'prompt' :     prompt,
        'rationale' : rationale,
        'medications_and_supplements' : medications_and_supplements,
        'is_met' : 1 if is_met else 0,
        'confidence' : str(confidence),
        'true_label' : patient['labels'][criterion],
        'completion_tokens' : stat.completion_tokens if stat is not None else None,
        'prompt_tokens' :     stat.prompt_tokens     if stat is not None else None,
    }



def generate_result_koopman(query, assessment: CriterionAssessment, stat: UsageStat) -> Dict[str, Any]:
    """Helper method that constructs a result dictionary from a patient, note, prompt, and assessment."""
    
   
    patient  = query["patient"]
    trail    = query["trail"]
    trail_id = query["trail_id"]
    label    = query["label"]
    prompt   = query["prompt"]
     
     
    criterion:  str = assessment.criterion
    rationale:  str = assessment.rationale
    is_met:     str = assessment.is_met
    confidence: str = assessment.confidence
    medications_and_supplements: List[str] = assessment.medications_and_supplements
     
    return {
        'patient_id' : patient['patient_id'],
        'trail_id':   trail_id,
        'note' :       patient["ehr"],
        'criterion' :  criterion,
        'prompt' :     prompt,
        'rationale' : rationale,
        'medications_and_supplements' : medications_and_supplements,
        'is_met' : 1 if is_met else 0,
        'confidence' : str(confidence),
        'true_label' : label,
        'completion_tokens' : stat.completion_tokens if stat is not None else None,
        'prompt_tokens' :     stat.prompt_tokens     if stat is not None else None,
    }
    
def prompt__one_criteria(patient: Dict[str, Any], note: str, criterion: str, definition: str, is_exclude_rationale: bool = False) -> str:
    """Given a clinical note, specific criterion, and that criterion's definition, constructs a prompt for the LLM."""
    prompt: str = f"""

# Task
Your job is to decide whether the given patient meets the inclusion criterion.

# Patient

Below is a clinical note describing the patient's current health status:

```
{note}
```

# Current Date

Assume that the current date is: {XMLDataLoader.get_current_date_for_patient(patient)}

# Inclusion Criterion

The inclusion criterion being assessed is: "{criterion}: {definition}"

# Assessment

Given the criterion above, use the patient's clinical note to determine whether the patient meets this criterion. Think step by step, and justify your answer.

Format your response as a JSON object with the following keys: 
* criterion: str - The name of the criterion being assessed
* medications_and_supplements: List[str] - The names of all current medications and supplements that the patient is taking
{'* rationale: str - Your reasoning as to why the patient does or does not meet that criterion' if not is_exclude_rationale else ''}
* is_met: bool - "true" if the patient meets that criterion, or it can be inferred that they meet that criterion with common sense. "false" if the patient does not or it is impossible to assess this given the provided information.
* confidence: str - Either "low", "medium", or "high" to reflect your confidence in your response

An example of how your JSON response should be formatted is shown below:
```json
{{
    "criterion" : "{criterion}",
    "medications_and_supplements" : [ "medication_1", "medication_2"],
    {'"rationale" : "something something",' if not is_exclude_rationale else ''}
    "is_met" : true/false,
    "confidence" : "low/medium/high",
}}
```
The above example is only for illustration purposes only. It does not reflect the actual criterion or patient for this task.

Please provide your response:
"""
    return prompt

Query: namedtuple = namedtuple('Query', ['patient', 'note', 'note_idx', 'criterion', 'definition', 'prompt'])
null_assessment: namedtuple = namedtuple('null_assessment', ['criterion', 'rationale', 'is_met', 'confidence', 'medications_and_supplements'])
null_assessments: namedtuple = namedtuple('null_assessments', ['assessments'])

def pipeline(dataloader: XMLDataLoader, 
            patients: List[Dict[str, Any]],
            patient_2_criterion_2_docs: Dict[str, Dict[str, List]],
            llm_model: str,
            llm_kwargs: Dict,
            is_all_criteria: bool = False,
            is_all_notes: bool = False,
            is_chunk_keep_full_note: bool = False,
            is_exclude_rationale: bool = False) -> Tuple[List[Dict[str, Any]], List[UsageStat]]:
    """For each criterion, look at `each / all notes at once` and query the LLM to asses if match or not."""
    results: List[Dict[str, Any]] = []
    stats: List[UsageStat] = []
    queries: List[namedtuple] = []

    # Count tokens in prompt for truncation
    if 'tokenizer' in llm_kwargs:
        if is_all_criteria:
            n_tokens_in_prompt: int = len(llm_kwargs['tokenizer'].encode(prompt__all_criteria(patients[0], '', dataloader.definitions, is_exclude_rationale=is_exclude_rationale)))
        else:
            n_tokens_in_prompt: int = len(llm_kwargs['tokenizer'].encode(prompt__one_criteria(patients[0], '', dataloader.criteria[0], dataloader.definitions[dataloader.criteria[0]], is_exclude_rationale=is_exclude_rationale)))

    criteria: List[str] = [ 'all' ] if is_all_criteria else dataloader.criteria
    for patient in patients:
        criterion_2_docs: Dict[str, str] = patient_2_criterion_2_docs[patient['patient_id']]
        for criterion in tqdm(criteria, desc='Looping through criteria...', leave=False):
            if is_chunk_keep_full_note:
                # Get full notes corresponding to returned chunks
                if criterion == 'all':
                    note_idxs: List[int] = list(set([ x['metadata']['note_idx'] for criterion in criterion_2_docs.keys() for x in criterion_2_docs[criterion] ]))
                    docs: List[str] = [ patient['ehr'][x] for x in note_idxs ]
                else:
                    note_idxs: List[int] = list(set([ x['metadata']['note_idx'] for x in criterion_2_docs[criterion] ]))
                    docs: List[str] = [ patient['ehr'][x] for x in note_idxs ]
            else:
                # Only keep returned chunks
                if criterion == 'all':
                    note_idxs: List[int] = [ x['metadata']['note_idx'] for criterion in criterion_2_docs.keys() for x in criterion_2_docs[criterion] ]
                    docs: List[str] = [ x['text'] for criterion in criterion_2_docs.keys() for x in criterion_2_docs[criterion] ]
                else:
                    note_idxs: List[int] = [ x['metadata']['note_idx'] for x in criterion_2_docs[criterion] ]
                    docs: List[str] = [ x['text'] for x in criterion_2_docs[criterion] ]

            # Check that there are no duplicates
            assert len(docs) == len(set(docs)), f"Error - duplicates found {len(docs)} != {len(set(docs))}"
            
            # If we're looking at all notes at once, then concatenate all notes into one big note
            if is_all_notes:
                doc_sep: str = '******************' # need to separate docs so that LLM doesn't get confused
                docs = [ f"\n{doc_sep}\n".join(docs) ]
                note_idxs = [ note_idxs ]

            for doc, note_idx in zip(docs, note_idxs):
                if is_all_criteria:
                    if 'tokenizer' in llm_kwargs and model_2_tokens[llm_model] < 32_000:
                        doc = truncate_doc(llm_kwargs['tokenizer'], doc, max_tokens=model_2_tokens[llm_model] - n_tokens_in_prompt - 64) # buffer of 50 tokens
                    prompt: str = prompt__all_criteria(patient, doc, dataloader.definitions, is_exclude_rationale=is_exclude_rationale)
                    queries.append(Query(patient=patient, note=doc, note_idx=note_idx, criterion=None, definition=None, prompt=prompt))
                else:
                    prompt: str = prompt__one_criteria(patient, doc, criterion, dataloader.definitions[criterion], is_exclude_rationale=is_exclude_rationale)
                    queries.append(Query(patient=patient, note=doc, note_idx=note_idx, criterion=criterion, definition=dataloader.definitions[criterion], prompt=prompt))

    # Sanity checks
    if is_all_criteria and is_all_notes:
        assert len(queries) == len(patients), f"Expected {len(patients)} queries, but got {len(queries)} instead."
    elif is_all_criteria and is_chunk_keep_full_note:
        assert len(queries) == sum([ len(x['ehr']) for x in patients ]), f"Expected {sum([ len(x['ehr']) for x in patients])} queries, but got {len(queries)} instead."
    elif is_all_notes:
        assert len(queries) == len(criteria) * len(patients), f"Expected {len(criteria) * len(patients)} queries, but got {len(queries)} instead."

    # Query LLM whether it satisfies the criteria
    if 'openai_client' in llm_kwargs:
        responses: List[Tuple] = batch_query_openai([ x.prompt for x in queries ], llm_model, 'CriterionAssessments' if is_all_criteria else 'CriterionAssessment')
    else:
        responses: List[Tuple] = batch_query_hf([ x.prompt for x in queries ], llm_model, 'CriterionAssessments' if is_all_criteria else 'CriterionAssessment', llm_kwargs)
    assert len(responses) == len(queries), f"Expected {len(queries)} responses, but got {len(responses)} instead."

    # Process each response
    for (response, stat), query in zip(responses, queries):
        is_response_null: bool = response is None
        if is_response_null:
            if is_all_criteria:
                response = null_assessments(assessments=[null_assessment(criterion=criterion, rationale=None, is_met=None, confidence=None, medications_and_supplements=[]) for criterion in dataloader.criteria])
            else:
                response = null_assessment(criterion=query.criterion, rationale=None, is_met=None, confidence=None, medications_and_supplements=[])

        assessments: List[CriterionAssessment] = response.assessments if is_all_criteria else [ response ]
        
        # Parse response
        stats.append(stat)
        for assessment_idx, assessment in enumerate(assessments):
            if not is_all_criteria and not is_response_null:
                # Force (in case GPT responds with `criterion:definition` as the criterion)
                assessment.criterion = query.criterion
            if assessment.criterion not in dataloader.criteria:
                # Error - criterion {assessment.criterion} not in dataloader.criteria, so we can't use it
                continue
            if assessment.criterion in [ x.criterion for x in assessments[:assessment_idx] ]:
                # Ignore duplicated criterion
                continue
            results.append(generate_result(query.patient, query.note, query.note_idx, query.prompt, assessment, stat))
        
        # Backfill any missing criteria
        if is_all_criteria and not is_response_null:
            for criterion in dataloader.criteria:
                if criterion not in [ x.criterion for x in assessments ]:
                    results.append(generate_result(query.patient, query.note, query.note_idx, query.prompt, null_assessment(criterion=criterion, rationale=None, is_met=None, confidence=None, medications_and_supplements=[]), stat))
    return results, stats



def pipeline_koopman(dataset, 
            patient: List[Dict[str, Any]],
            llm_model: str,
            llm_kwargs: Dict,
            is_exclude_rationale: bool = False) -> Tuple[List[Dict[str, Any]], List[UsageStat]]:
    
    """For each criterion, look at `each / all notes at once` and query the LLM to asses if match or not."""
    results: List[Dict[str, Any]] = []
    stats:   List[UsageStat] = []
    queries: List[namedtuple] = []
    
    
    for label in tqdm(patient["labels"]):
        trail       =  dataset["trails"][label["trail"]]
        
        prompt,n_criterias = prompt__all_criteria_koopman(patient,trail,is_exclude_rationale=is_exclude_rationale)
        queries.append({"patient":patient,
                        "trail":trail,
                        "trail_id":label["trail"],
                        "prompt":prompt,
                        "label":label["label"],
                        "total_criterion":n_criterias})
    
    # for debugging
    #queries = [queries[0]]
    #import pdb;pdb.set_trace()

    # Query LLM whether it satisfies the criteria
    if 'openai_client' in llm_kwargs:
        responses: List[Tuple] = batch_query_openai([ x["prompt"] for x in queries ], llm_model, 'CriterionAssessments')
        
    else:
        responses: List[Tuple] = batch_query_hf([ x["prompt"] for x in queries ], llm_model, 'CriterionAssessments')
        
    #assert len(responses) == len(queries), f"Expected {len(queries)} responses, but got {len(responses)} instead."
    
    # response is type 'CriterionAssessments'
    
    
    # Process each response
    for (response, stat), query in zip(responses, queries):
        is_response_null: bool = response is None
        
        if is_response_null:
            response = null_assessments(assessments=[null_assessment(criterion =f"criterion_{criterion}", 
                                                                     rationale =None, 
                                                                     is_met     =None, 
                                                                     confidence =None, 
                                                                     medications_and_supplements=[]) for criterion in range(0,queryn_criterias["total_criterion"])])
            
        assessments: List[CriterionAssessment] = response.assessments 
        stats.append(stat)
     
        for assessment_idx, assessment in enumerate(assessments):
            results.append(generate_result_koopman(query, assessment, stat))
        
       
    return results, stats


def prompt__all_criteria(patient: Dict[str, Any], note: str, criteria_2_definition: Dict[str, str], is_exclude_rationale: bool = False) -> str:
    """Given a clinical note and all criteria, constructs a prompt for the LLM."""
    section_criteria: str = "\n".join([ f"- {criterion}: {definition}" for criterion, definition in criteria_2_definition.items() ])
    prompt: str = f"""

# Task
Your job is to decide which of the following inclusion criteria the given patient meets.

# Patient

Below is a clinical note describing the patient's current health status:

```
{note}
```

# Current Date

Assume that the current date is: {XMLDataLoader.get_current_date_for_patient(patient)}

# Inclusion Criteria

The inclusion criteria being assessed are listed below, followed by their definitions: 
{section_criteria}

# Assessment

For each of the criteria above, use the patient's clinical note to determine whether the patient meets each criteria. Think step by step, and justify your answer.

Format your response as a JSON list of dictionaries, where each dictionary contains the following elements:
* criterion: str - The name of the criterion being assessed
* medications_and_supplements: List[str] - The names of all current medications and supplements that the patient is taking
{'* rationale: str - Your reasoning as to why the patient does or does not meet that criterion' if not is_exclude_rationale else ''}
* is_met: bool - "true" if the patient meets that criterion, or it can be inferred that they meet that criterion with common sense. "false" if the patient does not or it is impossible to assess this given the provided information.
* confidence: str - Either "low", "medium", or "high" to reflect your confidence in your response

An example of how your JSON response should be formatted is shown below, where the list of JSON dictionaries is stored in the "assessments" key:
```json
{{ 
    "assessments" : [
        {{
            "criteria_1" : "something",
            "medications_and_supplements" : [ "medication_1", "medication_2"],
            {'"rationale" : "something something",' if not is_exclude_rationale else ''}
            "is_met" : true/false,
            "confidence" : "low/medium/high",
        }},
        {{
            "criteria_2" : "something",
            "medications_and_supplements" : [],
            {'"rationale" : "something something",' if not is_exclude_rationale else ''}
            "is_met" : true/false,
            "confidence" : "low/medium/high",
        }},
        ...
    ]
}}
```
The above example is only for illustration purposes only. It does not reflect the actual criteria or patient for this task.

Please analyze the given patient and inclusion criteria. Remember to include all inclusion criteria in your returned JSON dictionary. Please provide your JSON response:
"""
    return prompt



def prompt__all_criteria_koopman(patient: Dict[str, Any], trail: Dict[str, Any],  is_exclude_rationale: bool = False) -> str:
    """Given a clinical note and all criteria, constructs a prompt for the LLM."""
    
    inclusion_criteria: str = "\n".join([ f"- inclusion_criteria_{i}: {criteria}" for i,criteria in enumerate(trail["inclusion_criteria"]) ])
    exclusion_criteria: str = "\n".join([ f"- exclusion_criteria_{i}: {criteria}" for i,criteria in enumerate(trail["exclusion_criteria"]) ])
    
    prompt: str = f"""
# Task
Your job is to indicate which of the following inclusion and exlusion  criteria is met by the patient

# Patient

Below is a clinical note describing the patient's current health status:

```
{patient["ehr"]}
```

# Inclusion Criteria

The inclusion criteria being assessed are listed below, followed by their definitions: 
{inclusion_criteria}


The exclusion criteria being assessed are listed below, followed by their definitions: 
{exclusion_criteria}

# Assessment

For each of the criteria above (both inclusion and exclusion), use the patient's clinical note to determine whether the patient meets each criteria. Think step by step, and justify your answer.

Format your response as a JSON list of dictionaries, where each dictionary contains the following elements:
* criterion: str - The name of the criterion being assessed
* medications_and_supplements: List[str] - The names of all current medications and supplements that the patient is taking
{'* rationale: str - Your reasoning as to why the patient does or does not meet that criterion' if not is_exclude_rationale else ''}
* is_met: bool - "true" if the patient meets that criterion, or it can be inferred that they meet that criterion with common sense. "false" if the patient does not or it is impossible to assess this given the provided information.
* confidence: str - Either "low", "medium", or "high" to reflect your confidence in your response

An example of how your JSON response should be formatted is shown below, where the list of JSON dictionaries is stored in the "assessments" key:
```json
{{ 
    "assessments" : [
        {{
            "criteria_1" : "something",
            "medications_and_supplements" : [ "medication_1", "medication_2"],
            {'"rationale" : "something something",' if not is_exclude_rationale else ''}
            "is_met" : true/false,
            "confidence" : "low/medium/high",
        }},
        {{
            "criteria_2" : "something",
            "medications_and_supplements" : [],
            {'"rationale" : "something something",' if not is_exclude_rationale else ''}
            "is_met" : true/false,
            "confidence" : "low/medium/high",
        }},
        ...
    ]
}}
```
The above example is only for illustration purposes only. It does not reflect the actual criteria or patient for this task.

Please analyze the given patient and  inclusion and exclusion criteria. Remember to include all  inclusion and exclusion criteria in your returned JSON dictionary. Please provide your JSON response:
"""
    return prompt, len(inclusion_criteria) + len(exclusion_criteria)



