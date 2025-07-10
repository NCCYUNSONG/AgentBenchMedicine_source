import argparse
import re, random, time, json, os
import configparser
import openai

# Load configuration from config.ini
def load_config():
    config = configparser.ConfigParser()
    try:
        with open("config.ini", "r") as f:
            config.read_file(f)
        return config
    except FileNotFoundError:
        raise Exception("config.ini not found. Please create it with model configurations.")

config = load_config()

# Initialize custom OpenAI clients
CUSTOM_OPENAI_CLIENTS = {
    "llama4_maverick": openai.OpenAI(base_url=config["llama4_maverick"]["base_url"], api_key=config["llama4_maverick"]["api_key"]),
    "gpt-4o-mini": openai.OpenAI(base_url=config["gpt-4o-mini"]["base_url"], api_key=config["gpt-4o-mini"]["api_key"]),
    "gemma": openai.OpenAI(base_url=config["gemma"]["base_url"],api_key=config["gemma"]["api_key"]),
    "gpt-4.1": openai.OpenAI(base_url=config["gpt-4.1"]["base_url"],api_key=config["gpt-4.1"]["api_key"]),
}

CUSTOM_MODEL_NAMES = {
    "llama4_maverick": config["llama4_maverick"]["model"],
    "gpt-4o-mini": config["gpt-4o-mini"]["model"],
    "gemma": config["gemma"]["model"],
    "gpt-4.1": config["gpt-4.1"]["model"],
}

token_usage_counter = {
    "llama4_maverick": 0,
    "gpt-4o-mini": 0,
    "gemma": 0,
    "gpt-4.1": 0
}

def query_model(model_str, prompt, system_prompt, tries=30, timeout=20.0, image_requested=False, scene=None,
                max_prompt_len=2 ** 14, clip_prompt=False):
    supported_models = ["llama4_maverick", "gpt-4o-mini", "gpt-4.1", "gemma"]
    if model_str not in supported_models:
        raise Exception(f"No model by the name {model_str}")

    logs = []
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(tries):
        if clip_prompt:
            prompt = prompt[:max_prompt_len]
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            if image_requested and hasattr(scene, "image_url") and scene.image_url:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": prompt},
                         {"type": "image_url",
                          "image_url": {
                              "url": scene.image_url
                          }
                         }
                     ]}
                ]

            client = CUSTOM_OPENAI_CLIENTS[model_str]
            model_name = CUSTOM_MODEL_NAMES[model_str]
            max_tokens = 512
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else len(prompt.split()),
                "completion_tokens": response.usage.completion_tokens if response.usage else len(answer.split()),
                "total_tokens": response.usage.total_tokens if response.usage else len(prompt.split()) + len(answer.split())
            }
            token_usage_counter[model_str] += token_usage["total_tokens"]

            answer = re.sub(r"\s+", " ", answer)
            logs.append(f"Attempt {attempt + 1} succeeded")
            return answer, logs, token_usage

        except Exception as e:
            logs.append(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(timeout)
            continue
    raise Exception(f"Max retries exceeded: {logs[-1]}")

class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("agentclinic/agentclinic_medqa.jsonl", "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioMedQAExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMedQAExtended:
    def __init__(self) -> None:
        with open("agentclinic/agentclinic_medqa_extended.jsonl", "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMedQAExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioMIMICIVQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.tests = scenario_dict["OSCE_Examination"]["Test_Results"]
        self.diagnosis = scenario_dict["OSCE_Examination"]["Correct_Diagnosis"]
        self.patient_info = scenario_dict["OSCE_Examination"]["Patient_Actor"]
        self.examiner_info = scenario_dict["OSCE_Examination"]["Objective_for_Doctor"]
        self.physical_exams = scenario_dict["OSCE_Examination"]["Physical_Examination_Findings"]

    def patient_information(self) -> dict:
        return self.patient_info

    def examiner_information(self) -> dict:
        return self.examiner_info

    def exam_information(self) -> dict:
        exams = self.physical_exams
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> dict:
        return self.diagnosis

class ScenarioLoaderMIMICIV:
    def __init__(self) -> None:
        with open("agentclinic/agentclinic_mimiciv.jsonl", "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioMIMICIVQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioNEJMExtended:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"] for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis

class ScenarioLoaderNEJMExtended:
    def __init__(self) -> None:
        with open("agentclinic/agentclinic_nejm_extended.jsonl", "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJMExtended(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class ScenarioNEJM:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.question = scenario_dict["question"]
        self.image_url = scenario_dict["image_url"]
        self.diagnosis = [_sd["text"] for _sd in scenario_dict["answers"] if _sd["correct"]][0]
        self.patient_info = scenario_dict["patient_info"]
        self.physical_exams = scenario_dict["physical_exams"]

    def patient_information(self) -> str:
        patient_info = self.patient_info
        return patient_info

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> str:
        exams = self.physical_exams
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis

class ScenarioLoaderNEJM:
    def __init__(self) -> None:
        with open("agentclinic/agentclinic_nejm.jsonl", "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]
        self.scenarios = [ScenarioNEJM(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class PatientAgent:
    def __init__(self, scenario, backend_str="llama4_maverick", bias_present=None) -> None:
        self.disease = ""
        self.symptoms = ""
        self.agent_hist = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.logs = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.reset()

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you had a friend with similar symptoms to yours that was diagnosed with cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose this as cancer based on past cases. This affects how you interact with the doctor.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your friends who heard about your symptoms believe that you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "self_diagnosis":
            return "\nYou recently performed a thorough investigation of your symptoms online and have come to the conclusion that you should be diagnosed with cancer. While the doctor may believe otherwise you remain convinced you have cancer. This affects how you interact with the doctor.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your doctor because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your doctor because you find out that they went to a low ranked medical school and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your doctor because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your doctor because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_patient(self, question) -> str:
        answer, logs, tokens = query_model(
            self.backend,
            f"\nHere is a history of your dialogue: {self.agent_hist}\nHere was the doctor response: {question}\nNow please continue your dialogue\nPatient: ",
            self.system_prompt()
        )
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.logs.extend(logs)
        for key in self.token_usage:
            self.token_usage[key] += tokens.get(key, 0)
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = """You are a patient in a clinic who only responds in the form of dialogue. You are being inspected by a doctor who will ask you questions and will perform exams on you in order to understand your disease. Your answer will only be 1-3 sentences in length."""
        symptoms = "\n\nBelow is all of your information. {}. \n\n Remember, you must not reveal your disease explicitly but may only convey the symptoms you have in the form of dialogue if you are asked.".format(
            self.symptoms)
        return base + bias_prompt + symptoms

    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

class DoctorAgent:
    def __init__(self, scenario, backend_str="llama4_maverick", max_infs=20, bias_present=None, img_request=False) -> None:
        self.infs = 0
        self.MAX_INFS = max_infs
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.bias_present = (None if bias_present == "None" else bias_present)
        self.scenario = scenario
        self.logs = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.reset()
        self.img_request = img_request

    def generate_bias(self) -> str:
        if self.bias_present == "recency":
            return "\nRecently, you learned that a doctor with similar symptoms to the patient that diagnosed their patient with cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "frequency":
            return "\nYou recently learned that most frequently, doctors diagnose patients with similar symptoms with cancer based on past cases. This affects how you interact with the patient.\n"
        elif self.bias_present == "false_consensus":
            return "\nMost of your colleagues who heard about the patient's symptoms believe that they have cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "status_quo":
            return "\nYour hospital has been diagnosing these patient's symptoms with cancer for the past 10 years. This affects how you interact with the patient.\n"
        elif self.bias_present == "confirmation":
            return "\nYou are initially confident that the patient has cancer. This affects how you interact with the patient.\n"
        elif self.bias_present == "gender":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular gender and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "race":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular race and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "sexual_orientation":
            return "\nYou are uncomfortable with your patient because you find out that they are a particular sexual orientation and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "cultural":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular culture and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "education":
            return "\nYou are uncomfortable with your patient because you find out that they are uneducated and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "religion":
            return "\nYou are uncomfortable with your patient because you find out that they believe in a particular religion and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present == "socioeconomic":
            return "\nYou are uncomfortable with your patient because you find out that they are from a particular socioeconomic background and you do not trust their judgement. This affects how you interact with them.\n"
        elif self.bias_present is None:
            pass
        else:
            print("BIAS TYPE {} NOT SUPPORTED, ignoring bias...".format(self.bias_present))
        return ""

    def inference_doctor(self, question, image_requested=False) -> str:
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"
        answer, logs, tokens = query_model(
            self.backend,
            f"\nHere is a history of your dialogue: {self.agent_hist}\nHere was the patient response: {question}\nNow please continue your dialogue\nDoctor: ",
            self.system_prompt(),
            image_requested=image_requested,
            scene=self.scenario
        )
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.infs += 1
        self.logs.extend(logs)
        for key in self.token_usage:
            self.token_usage[key] += tokens.get(key, 0)
        return answer

    def system_prompt(self) -> str:
        bias_prompt = ""
        if self.bias_present is not None:
            bias_prompt = self.generate_bias()
        base = "You are a doctor named Dr. Agent who only responds in the form of dialogue. You are inspecting a patient who you will ask questions in order to understand their disease. You are only allowed to ask {} questions total before you must make a decision. You have asked {} questions so far. You can request test results using the format \"REQUEST TEST: [test]\". For example, \"REQUEST TEST: Chest_X-Ray\". Your dialogue will only be 1-3 sentences in length. Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\"".format(
            self.MAX_INFS, self.infs)
        image_instruction = " If a medical image is provided, analyze it and incorporate its findings into your dialogue or diagnosis." if hasattr(self.scenario, "image_url") else ""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n Remember, you must discover their disease by asking them questions. You are also able to provide exams.".format(
            self.presentation)
        return base + bias_prompt + image_instruction + presentation

    def reset(self) -> None:
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

class MeasurementAgent:
    def __init__(self, scenario, backend_str="llama4_maverick") -> None:
        self.agent_hist = ""
        self.presentation = ""
        self.backend = backend_str
        self.scenario = scenario
        self.logs = []
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.reset()

    def inference_measurement(self, question) -> str:
        answer, logs, tokens = query_model(
            self.backend,
            f"\nHere is a history of the dialogue: {self.agent_hist}\nHere was the doctor measurement request: {question}",
            self.system_prompt()
        )
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        self.logs.extend(logs)
        for key in self.token_usage:
            self.token_usage[key] += tokens.get(key, 0)
        return answer

    def system_prompt(self) -> str:
        base = "You are a measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = "\n\nBelow is all of the information you have. {}. \n\n If the requested results are not in your data then you can respond with NORMAL READINGS.".format(
            self.information)
        return base + presentation

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"

    def reset(self) -> None:
        self.agent_hist = ""
        self.information = self.scenario.exam_information()

def compare_results(diagnosis, correct_diagnosis, moderator_llm):
    answer, logs, tokens = query_model(
        moderator_llm,
        f"\nHere is the correct diagnosis: {correct_diagnosis}\nHere was the doctor dialogue: {diagnosis}\nAre these the same?",
        "You are responsible for determining if the correct diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."
    )
    return answer.lower(), logs, tokens

def main(inf_type, doctor_bias, patient_bias, doctor_llm, patient_llm, measurement_llm,
         moderator_llm, num_scenarios, dataset, img_request, total_inferences):
    total_start_time = time.time()
    total_correct = 0
    total_presents = 0
    all_scenarios_data = []

    if dataset == "MedQA":
        scenario_loader = ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        scenario_loader = ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM":
        scenario_loader = ScenarioLoaderNEJM()
    elif dataset == "NEJM_Ext":
        scenario_loader = ScenarioLoaderNEJMExtended()
    elif dataset == "MIMICIV":
        scenario_loader = ScenarioLoaderMIMICIV()
    else:
        raise Exception(f"Dataset {dataset} does not exist")
    if num_scenarios is None: num_scenarios = scenario_loader.num_scenarios
    for _scenario_id in range(min(num_scenarios, scenario_loader.num_scenarios)):
        print(f"Starting scenario {_scenario_id + 1}/{num_scenarios}")
        start_time = time.time()
        total_presents += 1

        scenario = scenario_loader.get_scenario(id=_scenario_id)
        meas_agent = MeasurementAgent(scenario=scenario, backend_str=measurement_llm)
        patient_agent = PatientAgent(scenario=scenario, bias_present=patient_bias, backend_str=patient_llm)
        doctor_agent = DoctorAgent(scenario=scenario, bias_present=doctor_bias, backend_str=doctor_llm,
                                  max_infs=total_inferences, img_request=img_request)

        scenario_data = {
            "scenario_id": _scenario_id,
            "dataset": dataset,
            "doctor_llm": doctor_llm,
            "patient_llm": patient_llm,
            "measurement_llm": measurement_llm,
            "moderator_llm": moderator_llm,
            "doctor_bias": doctor_bias,
            "patient_bias": patient_bias,
            "inf_type": inf_type,
            "img_request": img_request,
            "total_inferences": total_inferences,
            "dialogue": [],
            "correct_diagnosis": scenario.diagnosis_information(),
            "doctor_diagnosis": None,
            "is_correct": None,
            "token_usage": {"doctor": {}, "patient": {}, "measurement": {}, "total": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        }

        send_img_first = dataset in ["NEJM", "NEJM_Ext"]

        if inf_type == "human_doctor":
            doctor_dialogue = input("\nQuestion for patient: ")
        else:
            doctor_dialogue = doctor_agent.inference_doctor("", image_requested=send_img_first)
        print(f"Doctor [0%]: {doctor_dialogue}")
        scenario_data["dialogue"].append({"role": "Doctor", "text": doctor_dialogue})

        pi_dialogue = ""
        for _inf_id in range(total_inferences):
            if _inf_id == total_inferences - 1:
                pi_dialogue += " This is the final question. Please provide a diagnosis."

            if "DIAGNOSIS READY" in doctor_dialogue:
                diagnosis_match = re.search(r"DIAGNOSIS READY: (.+)", doctor_dialogue)
                if diagnosis_match:
                    scenario_data["doctor_diagnosis"] = diagnosis_match.group(1)
                correctness, mod_logs, mod_tokens = compare_results(doctor_dialogue, scenario.diagnosis_information(), moderator_llm)
                scenario_data["is_correct"] = correctness == "yes"
                if scenario_data["is_correct"]:
                    total_correct += 1
                break

            if "REQUEST TEST" in doctor_dialogue:
                pi_dialogue = meas_agent.inference_measurement(doctor_dialogue)
                print(f"Measurement [{int(((_inf_id + 1) / total_inferences) * 100)}%]: {pi_dialogue}")
                scenario_data["dialogue"].append({"role": "Measurement", "text": pi_dialogue})
                meas_agent.logs = []
            else:
                if inf_type == "human_patient":
                    pi_dialogue = input("\nResponse to doctor: ")
                else:
                    pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
                print(f"Patient [{int(((_inf_id + 1) / total_inferences) * 100)}%]: {pi_dialogue}")
                scenario_data["dialogue"].append({"role": "Patient", "text": pi_dialogue})
                patient_agent.logs = []

            if dataset == "NEJM":
                if img_request:
                    imgs = "REQUEST IMAGES" in doctor_dialogue
                else:
                    imgs = True
            else:
                imgs = False

            if inf_type == "human_doctor":
                doctor_dialogue = input("\nQuestion for patient: ")
            else:
                doctor_dialogue = doctor_agent.inference_doctor(
                    pi_dialogue,
                    image_requested=imgs
                )

            print(f"Doctor [{int(((_inf_id + 1) / total_inferences) * 100)}%]: {doctor_dialogue}")
            scenario_data["dialogue"].append({"role": "Doctor", "text": doctor_dialogue})
            doctor_agent.logs = []

            scenario_data["token_usage"]["doctor"] = doctor_agent.token_usage
            scenario_data["token_usage"]["patient"] = patient_agent.token_usage
            scenario_data["token_usage"]["measurement"] = meas_agent.token_usage
            for key in scenario_data["token_usage"]["total"]:
                scenario_data["token_usage"]["total"][key] = (
                    doctor_agent.token_usage.get(key, 0) +
                    patient_agent.token_usage.get(key, 0) +
                    meas_agent.token_usage.get(key, 0)
                )
            end_time = time.time()
            scenario_data["time_consumption"] = end_time - start_time

        all_scenarios_data.append(scenario_data)
    total_end_time = time.time()

    output = {
        "simulation_settings": {
            "inf_type": inf_type,
            "doctor_bias": doctor_bias,
            "patient_bias": patient_bias,
            "doctor_llm": doctor_llm,
            "patient_llm": patient_llm,
            "measurement_llm": measurement_llm,
            "moderator_llm": moderator_llm,
            "dataset": dataset,
            "img_request": img_request,
            "total_inferences": total_inferences,
            "num_scenarios": num_scenarios
        },
        "results": all_scenarios_data,
        "summary": {
            "total_scenarios": total_presents,
            "total_correct": total_correct,
            "correct_rate": total_correct / total_presents if total_presents > 0 else 0,
            "total_token_usage": {
                "prompt_tokens": sum(s["token_usage"]["total"]["prompt_tokens"] for s in all_scenarios_data),
                "completion_tokens": sum(s["token_usage"]["total"]["completion_tokens"] for s in all_scenarios_data),
                "total_tokens": sum(s["token_usage"]["total"]["total_tokens"] for s in all_scenarios_data),
                "by_model": token_usage_counter
            }
        }
    }
    output["summary"]["total_time_seconds"] = total_end_time - total_start_time

    output_filename = f"simulation_results_{dataset}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"Results exported to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Medical Diagnosis Simulation CLI')
    parser.add_argument('--inf_type', type=str, choices=['llm', 'human_doctor', 'human_patient'], default='llm')
    parser.add_argument('--doctor_bias', type=str, help='Doctor bias type', default='None',
                        choices=["recency", "frequency", "false_consensus", "confirmation", "status_quo", "gender",
                                 "race", "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--patient_bias', type=str, help='Patient bias type', default='None',
                        choices=["recency", "frequency", "false_consensus", "self_diagnosis", "gender", "race",
                                 "sexual_orientation", "cultural", "education", "religion", "socioeconomic"])
    parser.add_argument('--doctor_llm', type=str, default='llama4_maverick', choices=['llama4_maverick', 'gpt-4o-mini', 'gemma', 'gpt-4.1'])
    parser.add_argument('--patient_llm', type=str, default='llama4_maverick', choices=['llama4_maverick', 'gpt-4o-mini', 'gemma', 'gpt-4.1'])
    parser.add_argument('--measurement_llm', type=str, default='llama4_maverick', choices=['llama4_maverick', 'gpt-4o-mini', 'gemma', 'gpt-4.1'])
    parser.add_argument('--moderator_llm', type=str, default='llama4_maverick', choices=['llama4_maverick', 'gpt-4o-mini', 'gemma', 'gpt-4.1'])
    parser.add_argument('--agent_dataset', type=str, default='MedQA')
    parser.add_argument('--doctor_image_request', type=bool, default=False)
    parser.add_argument('--num_scenarios', type=int, default=None, help='Number of scenarios to simulate')
    parser.add_argument('--total_inferences', type=int, default=20, help='Number of inferences between patient and doctor')

    args = parser.parse_args()

    main(args.inf_type, args.doctor_bias, args.patient_bias, args.doctor_llm, args.patient_llm,
         args.measurement_llm, args.moderator_llm, args.num_scenarios, args.agent_dataset,
         args.doctor_image_request, args.total_inferences)