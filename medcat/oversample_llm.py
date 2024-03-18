import os
from ctransformers import (
    AutoModelForCausalLM as cAutoModelForCausalLM,
    AutoTokenizer as cAutoTokenizer,
)

model = cAutoModelForCausalLM.from_pretrained(
            model_path_or_repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            model_file="mistral-7b-instruct-v0.2.Q5_K_M.gguf",
            model_type="mistral",
            hf=True,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.2,
            context_length=8096,
            max_new_tokens=2048,
            threads=os.cpu_count(),
            stream=True,
            gpu_layers=0
            )
tokenizer = cAutoTokenizer.from_pretrained(model)

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.8,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=5000,
)
mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# For Experiencer

prompt_template = """
### INSTRUCTION: Act as a synthetic data generation expert. Generate synthetic examples for a multiclass classification task where the goal is to determine if a medical entity mentioned in a sentence is for the patient, or the family or other. Generate 20 words per example. Generate 30 examples.
For class 2 (patient), generate diverse examples where the context strongly suggests that the medical entity is for the patient. For class 1 (family), generate examples where the context clearly indicates that medical entity is for the family. For class 0 (other), generate examples where the context clearly indicates that medical entity is for others. 

OUTPUT: Generate diverse examples, ensure coherence and naturalness of the generated sentences. Output format: 'tokens','medical entity (present in tokens)','class label'.

INPUT: The instances of class 0, 1, 2 are provided in the format: 'tokens','medical entity','class label'. Here are the examples INSTANCE1 INSTANCE2 INSTANCE3 INSTANCE4 INSTANCE5 INSTANCE6

### INSTANCE1:
{instance1}

### INSTANCE2:
{instance2}

### INSTANCE3:
{instance3}

### INSTANCE4:
{instance4}

### INSTANCE5:
{instance5}

### INSTANCE6:
{instance6}

 ### Answer:
 """

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["instance1","instance2","instance3","instance4","instance5"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)

input_text = {

    "instance1": "'The patient denied history of heartburn and/or gastroesophageal reflux disorder.  A modified barium swallow study was ordered to objectively evaluate the patient's swallowing function and safety and to rule out aspiration.', 'gastroesophageal reflux disorder', '1' ",

    "instance2": "'HISTORY: , A 59-year-old male presents in followup after being evaluated and treated as an in-patient by Dr. X for acute supraglottitis with airway obstruction and parapharyngeal cellulitis and peritonsillar cellulitis, admitted on 05/23/2008, discharged on 05/24/2008.', 'airway obstruction', '0'",

    "instance3": "'She has had some decrease in her appetite, although her weight has been stable.  She has had no fever, chills, or sweats.  Activity remains good and she has continued difficulty with depression associated with type 1 bipolar disease.  She had a recent CT scan of the chest and abdomen.  The report showed the following findings.  In the chest, there was a small hiatal hernia and a calcification in the region of the mitral valve.', 'depression' ,'0'",

    "instance4": "'PAST MEDICAL HISTORY:,1.  Hypertension.,2.  Depression.,3.  Myofascitis of the feet.,4.  Severe osteoarthritis of the knee.,5.  Removal of the melanoma from the right thigh in 1984.,6.  Breast biopsy in January of 1997, which was benign.,7.  History of Holter monitor showing ectopic beat.  Echocardiogram was normal.' , 'osteoarthritis of the knee' , '0'",

    "instance5": "'Skin and Lymphatics:  Examination of the skin does not reveal any additional scars, rashes, spots or ulcers.  No significant lymphadenopathy noted.,Spine:  Examination shows lumbar scoliosis with surgical scar with no major tenderness.' , 'lymphadenopathy' , '1'",

    'instance6': "' She has no chest pain, palpitations, coughing or wheezing.  She does get shortness of breath, no hematuria, dysuria, arthralgias, myalgias, rashes, jaundice, bleeding or clotting disorders.  The rest of the system review is negative as per the HPI.','clotting disorders','1'"

}

# Running generation on loop for multiple examples to be generated
llm_generations = []
for i in range(10):
    response = llm_chain(input_text)
    llm_generations.append([response['text']])
    print(f"Finished {i}th run")

