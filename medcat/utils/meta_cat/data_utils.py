from typing import Dict, Optional, Tuple, Iterable, List
from medcat.tokenizers_med.meta_cat_tokenizers import TokenizerWrapperBase


def prepare_from_json(data: Dict,
                      cntx_left: int,
                      cntx_right: int,
                      tokenizer: TokenizerWrapperBase,
                      cui_filter: Optional[set] = None,
                      replace_center: Optional[str] = None,
                      prerequisites: Dict = {},
                      lowercase: bool = True) -> Dict:
    """Convert the data from a json format into a CSV-like format for training. This function is not very efficient (the one
    working with spacy documents as part of the meta_cat.pipe method is much better). If your dataset is > 1M documents think
    about rewriting this function - but would be strange to have more than 1M manually annotated documents.

    Args:
        data (dict):
            Loaded output of MedCATtrainer. If we have a `my_export.json` from MedCATtrainer, than data = json.load(<my_export>).
        cntx_left (int):
            Size of context to get from the left of the concept
        cntx_right (int):
        cntx_right (int):
            Size of context to get from the right of the concept
        tokenizer (medcat.tokenizers.meta_cat_tokenizers):
            Something to split text into tokens for the LSTM/BERT/whatever meta models.
        replace_center (Optional[str]):
            If not None the center word (concept) will be replaced with whatever this is.
        prerequisites (Dict):
            A map of prerequisities, for example our data has two meta-annotations (experiencer, negation). Assume I want to create
            a dataset for `negation` but only in those cases where `experiencer=patient`, my prerequisites would be:
                {'Experiencer': 'Patient'} - Take care that the CASE has to match whatever is in the data. Defaults to `{}`.
        lowercase (bool):
            Should the text be lowercased before tokenization. Defaults to True.

    Returns:
        out_data (dict):
            Example: {'category_name': [('<category_value>', '<[tokens]>', '<center_token>'), ...], ...}
    """
    out_data: Dict = {}
    cnt_print = 0
    for project in data['projects']:
        for document in project['documents']:
            text = str(document['text'])
            if lowercase:
                text = text.lower()

            if len(text) > 0:

                doc_text = tokenizer(text)

                for ann in document.get('annotations', document.get('entities',
                                                                    {}).values()):  # A hack to suport entities and annotations
                    cui = ann['cui']
                    skip = False
                    if 'meta_anns' in ann and prerequisites:
                        # It is possible to require certain meta_anns to exist and have a specific value
                        for meta_ann in prerequisites:
                            if meta_ann not in ann['meta_anns'] or ann['meta_anns'][meta_ann]['value'] != prerequisites[
                                meta_ann]:
                                # Skip this annotation as the prerequisite is not met
                                skip = True
                                break

                    if not skip and (cui_filter is None or not cui_filter or cui in cui_filter):
                        if ann.get('validated', True) and (
                                not ann.get('deleted', False) and not ann.get('killed', False)
                                and not ann.get('irrelevant', False)):
                            start = ann['start']
                            end = ann['end']

                            # Get the index of the center token
                            flag = 0
                            ctoken_idx = []

                            for ind, pair in enumerate(doc_text['offset_mapping']):
                                if start <= pair[0] or start <= pair[1]:
                                    if end <= pair[1]:
                                        ctoken_idx.append(ind)
                                        break
                                    else:
                                        flag = 1
                                if flag == 1:
                                    if end <= pair[1] or end <= pair[0]:
                                        break
                                    else:
                                        ctoken_idx.append(ind)

                                # if all(elem in ctoken_idx for elem in [25,26,27,28,29]):
                                #     print("PRINTING",start,end,text[start:end])

                            ind = 0
                            for ind, pair in enumerate(doc_text['offset_mapping']):
                                if start >= pair[0] and start < pair[1]:
                                    break

                            # if len(ctoken_idx)==0:
                                # for ind, pair in enumerate(doc_text['offset_mapping']):
                                #     if start == pair[0] + 2 or start == pair[0] - 2 or start == pair[1] + 2 or start == pair[1] - 2:
                                #         if end <= pair[1]:
                                #             ctoken_idx.append(ind)
                                #             break
                                #         else:
                                #             flag = 1
                                #     if flag == 1:
                                #         if end <= pair[1] or end <= pair[0]:
                                #             break
                                #         else:
                                #             ctoken_idx.append(ind)
                                #
                                #
                                # print("TOKEN",text[start:end])
                                # for ind, pair in enumerate(doc_text['offset_mapping']):
                                #     print(start,end,"    ",pair[0],pair[1])


                            _start = max(0, ctoken_idx[0] - cntx_left)
                            _end = min(len(doc_text['input_ids']), ind + 1 + cntx_right)

                            cpos = cntx_left + min(0, ind - cntx_left)
                            cpos_new = [ x - _start for x in ctoken_idx]

                            if any(elem < 0 for elem in cpos_new) and cnt_print==0:
                                print("negative found",cpos_new)
                                print(_start,ctoken_idx)

                                for ind, pair in enumerate(doc_text['offset_mapping']):
                                    print(ind,"-->",start,end,"    ",pair[0],pair[1])
                                cnt_print+=1

                            _end = min(len(doc_text['input_ids']), ctoken_idx[-1] + 1 + cntx_right)
                            # print(_start, _end)
                            tkns = doc_text['input_ids'][_start:_end]

                            if replace_center is not None:
                                if lowercase:
                                    replace_center = replace_center.lower()
                                for p_ind, pair in enumerate(doc_text['offset_mapping']):
                                    if start >= pair[0] and start < pair[1]:
                                        s_ind = p_ind
                                    if end > pair[0] and end <= pair[1]:
                                        e_ind = p_ind

                                ln = e_ind - s_ind
                                tkns = tkns[:cpos] + tokenizer(replace_center)['input_ids'] + tkns[cpos + ln + 1:]

                            # Backward compatibility if meta_anns is a list vs dict in the new approach
                            meta_anns = []
                            if 'meta_anns' in ann:
                                meta_anns = ann['meta_anns'].values() if type(ann['meta_anns']) == dict else ann[
                                    'meta_anns']

                            # If the annotation is validated
                            for meta_ann in meta_anns:
                                name = meta_ann['name']
                                value = meta_ann['value']

                                sample = [tkns, cpos_new, value]

                                if name in out_data:
                                    out_data[name].append(sample)
                                else:
                                    out_data[name] = [sample]
    return out_data


def encode_category_values(data: Dict, existing_category_value2id: Optional[Dict] = None,
                           category_undersample=None) -> Tuple:
    """Converts the category values in the data outputed by `prepare_from_json`
    into integere values.

    Args:
        data (Dict):
            Output of `prepare_from_json`.
        existing_category_value2id(Optional[Dict]):
            Map from category_value to id (old/existing).

    Returns:
        dict:
            New data with integers inplace of strings for categry values.
        dict:
            Map rom category value to ID for all categories in the data.
    """
    data = list(data)
    if existing_category_value2id is not None:
        category_value2id = existing_category_value2id
    else:
        category_value2id = {}

    category_values = set([x[2] for x in data])

    # for c in category_values:
    #     if c not in category_value2id:
    #         category_value2id[c] = len(category_value2id)
    print("category_values2id", category_value2id)

    # Ensuring that each label has data and checking for class imbalance

    label_data = {key: 0 for key in category_value2id}
    label_data_2 = {}
    for i in range(len(data)):
        # print("data[i][2]",data[i][2])
        # print("category_values2id", category_value2id)

        if data[i][2] in category_value2id:
            label_data[data[i][2]] = label_data[data[i][2]] + 1
    print("label_data", label_data)

    # if 0 in label_data.values():
    #
    #     # This means one or more labels have no data; removing the label(s)
    #
    #     for key_0 in keys_with_value_0:
    #         # Replacing the encoding value
    #         category_value2id[list(category_value2id.keys())[-1]] = category_value2id[key_0]
    #         del category_value2id[key_0]
    #         # Sorting the dict as per ids (values) to ensure label encoding is continuous
    #
    #         category_value2id = dict(sorted(category_value2id.items(), key=lambda item: item[1]))

    if 0 in label_data.values():
        category_value2id_ = {}
        keys_ls = [key for key, value in category_value2id.items() if value != 0]
        for k in keys_ls:
            category_value2id_[k] = len(category_value2id_)

        print(f"Labels found with 0 data; updates made\nFinal label encoding mapping:", category_value2id_)
        category_value2id = category_value2id_

    # Map values to numbers
    data_2 = []
    for i in range(len(data)):
        if data[i][2] in category_value2id:
            data[i][2] = category_value2id[data[i][2]]
            data_2.append(data[i])

    label_data_ = {v: 0 for v in category_value2id.values()}
    for i in range(len(data_2)):
        # print("data[i][2]",data_3[i][2])
        # print("category_values2id", category_value2id)
        if data_2[i][2] in category_value2id.values():
            label_data_[data_2[i][2]] = label_data_[data_2[i][2]] + 1
    print("Encoded label_data", label_data_)

    if category_undersample is None:
        min_lab_data = min(label_data.values())

    else:
        min_lab_data = label_data[category_undersample]

    print("min_lab_data", min_lab_data)

    data_3 = []
    label_data_counter = {v: 0 for v in category_value2id.values()}

    for sample in data_2:
        if label_data_counter[sample[-1]] < min_lab_data:
            data_3.append(sample)
            label_data_counter[sample[-1]] += 1

    label_data = {v: 0 for v in category_value2id.values()}
    for i in range(len(data_3)):
        # print("data[i][2]",data_3[i][2])
        # print("category_values2id", category_value2id)
        if data_3[i][2] in category_value2id.values():
            label_data[data_3[i][2]] = label_data[data_3[i][2]] + 1
    print("Updated label_data", label_data)
    return data_3, data_2, category_value2id


def json_to_fake_spacy(data: Dict, id2text: Dict) -> Iterable:
    """Creates a generator of fake spacy documents, used for running
    meta_cat pipe separately from main cat pipeline.

    Args:
        data(Dict):
            Output from cat formated as: {<id>: <output of get_entities, ...}.
        id2text(Dict):
            Map from document id to text of that document.

    Returns:
        Generator:
            Generator of spacy like documents that can be feed into meta_cat.pipe.
    """
    for id_ in data.keys():
        ents = data[id_]['entities'].values()

        doc = Doc(text=id2text[id_], id_=id_)
        doc.ents.extend([Span(ent['start'], ent['end'], ent['id']) for ent in ents])

        yield doc


class Empty(object):
    def __init__(self) -> None:
        pass


class Span(object):
    def __init__(self, start_char: str, end_char: str, id_: str) -> None:
        self._ = Empty()
        self.start_char = start_char
        self.end_char = end_char
        self._.id = id_  # type: ignore
        self._.meta_anns = None  # type: ignore


class Doc(object):
    def __init__(self, text: str, id_: str) -> None:
        self._ = Empty()
        self._.share_tokens = None  # type: ignore
        self.ents: List = []
        # We do not have overlapps at this stage
        self._ents = self.ents
        self.text = text
        self.id = id_
