import argparse
import json
import os
import time

import openai
from tqdm import tqdm

MAX_API_RETRY = 10000
REQ_TIME_GAP = 4

parser = argparse.ArgumentParser()
parser.add_argument("-q", "--question-file")
parser.add_argument("-a", "--answer-file-list", nargs="+", default=[])
parser.add_argument('-o', '--output', help='Output file (defaults to stdout)')
parser.add_argument("-m", "--eval-model", default="gpt-3.5-turbo-0301")
parser.add_argument("-k", "--k", type=int, default=3)
parser.add_argument("-b", "--bpc", type=int, default=1)

args = parser.parse_args()

if args.eval_model == "gpt-4":
    cost_per_promtp_token = 0.03 / 1000
    cost_per_completion_token = 0.06 / 1000
elif args.eval_model == "gpt-3.5-turbo-0301":
    cost_per_promtp_token = 2/ 10**6
    cost_per_completion_token = 2/ 10**6
else:
    raise ValueError("Invalid evaluator name")


os.environ["OPENAI_API_KEY"] = "sk-" 
openai.api_key = os.environ["OPENAI_API_KEY"]

def gen_prompt(ques, ans1, ans2):
    sys_prompt = 'You are a helpful and precise assistant for checking the quality of the answer.'
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n"
    default_prompt =  """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. 

    Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
    Please first provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment. 
    Then, output two lines indicating the scores for Assistant 1 and 2, respectively.

    Output with the following format:
    Evaluation evidence: <your evluation explanation here>
    Score of the Assistant 1: <score>
    Score of the Assistant 2: <score>"""
    return sys_prompt, prompt_template.format(question=ques, answer_1=ans1, answer_2=ans2, prompt=default_prompt)

def query_gpt(system_prompt, uer_prompt):
    for i in range(MAX_API_RETRY):
        try:
            response = openai.ChatCompletion.create(
                model=args.eval_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": uer_prompt},
                ],
                temperature=1,
                max_tokens=512,
                n=args.k
            )
            return response
        except openai.error.RateLimitError:
            print('rate limit')
            time.sleep(30)
        except Exception as e:
            print('error')
    raise RuntimeError(f"Failed after {MAX_API_RETRY} retries.")


def get_eval(ques, ans1, ans2):
    cost = 0
    system_prompt, user_prompt = gen_prompt(ques, ans1, ans2)
    response = query_gpt(system_prompt, user_prompt)
    cost += response['usage']['prompt_tokens'] * cost_per_promtp_token
    cost += response['usage']['completion_tokens'] * cost_per_completion_token
    all_scores = []
    contents = []
    contents_bpc = []
    for choice in response["choices"]:
        content = choice["message"]["content"]
        score1, score2 = parse_score_from_review(content)
        if score1 == -1 or score2 == -1:
            continue
        all_scores.append([score1, score2])
        contents.append(content)
    
    if args.bpc == 1:
        system_prompt, user_prompt_bpc = gen_prompt(ques, ans2, ans1)
        response_bpc = query_gpt(system_prompt, user_prompt_bpc)
        cost += response_bpc['usage']['prompt_tokens'] * cost_per_promtp_token
        cost += response_bpc['usage']['completion_tokens'] * cost_per_completion_token
        for choice in response_bpc["choices"]:
            content = choice["message"]["content"]
            score2, score1 = parse_score_from_review(content)
            if score1 == -1 or score2 == -1:
                continue
            all_scores.append([score1, score2])
            contents_bpc.append(content)
    
    score1 = sum([score[0] for score in all_scores]) / len(all_scores)
    score2 = sum([score[1] for score in all_scores]) / len(all_scores)
    return contents, contents_bpc, cost, [score1, score2]


def parse_score_from_review(review):
    try:
        score1 = review.split("\n")[-2]
        score2 = review.split("\n")[-1]
        score1 = score1.split(":")[-1].strip()
        score2 = score2.split(":")[-1].strip()
        return [float(score1), float(score2)]
    except:
        print(f'Failed to parse scores from {review}')
        return [-1, -1]

def get_json_list(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "r") as f:
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
        return json_list


if __name__ == "__main__":
    
    question_jsons = get_json_list(args.question_file)
    answer1_jsons = get_json_list(args.answer_file_list[0])
    answer2_jsons = get_json_list(args.answer_file_list[1])
    
    assert len(question_jsons) == len(answer1_jsons) == len(answer2_jsons)
    
    reviews = []
    total_len = len(question_jsons)
    question_idx_list = list(range(total_len))
    
    for i in tqdm(question_idx_list):
        assert (
            answer1_jsons[i]["question_id"]
            == question_jsons[i]["question_id"]
            == answer2_jsons[i]["question_id"]
        )

        ques = question_jsons[i]["text"]
        ans1 = answer1_jsons[i]["text"]
        ans2 = answer2_jsons[i]["text"]
        
        reviews.append(get_eval(ques, ans1, ans2))
        
        # To avoid the rate limit set by OpenAI
        time.sleep(REQ_TIME_GAP)

    total_cost = 0
    model1_vs_model2 = {
        'win': 0,
        'tie': 0,
        'loss': 0
    }
    with open(f"{args.output}", "w") as output_review_file:
        for idx, (contents, contents_bpc, cost, [score1, score2]) in enumerate(reviews):
            results = {
                "question_id": question_jsons[idx]["question_id"],
                "question": question_jsons[idx]["text"],
                "review": contents,
                "review_bpc": contents_bpc,
                "score": [score1, score2],
            }
            output_review_file.write(json.dumps(results) + "\n")
            total_cost += cost 
            
            if score1 == score2:
                model1_vs_model2['tie'] += 1
                
            elif score1 > score2:
                model1_vs_model2['win'] += 1
            else:
                model1_vs_model2['loss'] += 1
    
    print(f'Evaluation results (model1_vs_model2):\n{model1_vs_model2}')
    print(f'Evaluation cost: ${total_cost:.2f}.')

