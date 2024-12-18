import logging
import os
import json


def get_logger(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    log_file = os.path.join(log_dir, "train.log")

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def apply_chat_template(conversation):
    return "\n\n".join(
        f"{'Human' if qa['role'] == 'user' else 'Assistant'}: {qa['content']}" for qa in conversation)


# def write_to_json(output_dir, model_name, ratio_or_length, seed, metrics):
#     if not os.path.exists(output_dir):
#         data = {}
#         print(f"File not found. Initializing a new file at {output_dir}.")
#     else:
#         with open(output_dir, 'r') as f:
#             data = json.load(f)
#
#     if model_name not in data:
#         data[model_name] = {}
#
#     # 检查answer_length是否存在
#     if ratio_or_length not in data[model_name]:
#         data[model_name][ratio_or_length] = {}
#
#     # 检查seed是否存在
#     if seed not in data[model_name][ratio_or_length]:
#         data[model_name][ratio_or_length][seed] = {}
#
#     for k, v in metrics.items():
#         data[model_name][ratio_or_length][seed][k] = v
#
#     with open(output_dir, 'w') as f:
#         json.dump(data, f, indent=4)
#
#     print(f"Evaluation results for {seed} of {model_name} updated.")
