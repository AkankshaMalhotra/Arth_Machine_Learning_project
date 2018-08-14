from flask import Flask, render_template, request
import random
from Arth import Arth
import io


app = Flask(__name__)



def shuffle(q):
    """
    This function is for shuffling
    the dictionary elements.
    """
    selected_keys = []
    count = 0
    while count < len(q):
        current_selection = random.choice(list(q.keys()))
        if current_selection not in selected_keys:
            selected_keys.append(current_selection)
            count += 1
    return selected_keys


@app.route('/input')
def input():
    return render_template('input.html')

ques = {}
answers = {}
ans = {}
questions = {}
rev_index = {}
correct ={}
q = 0
clusters = 0

@app.route('/quiz', methods=['POST', 'GET'])
def quiz():
    result = request.form
    file_name = result["text"]
    with io.open(file_name, "r",  encoding="utf-8") as g:
        text = g.read()
    global q
    q = Arth(text)
    global clusters
    data, clusters = q.word_gen()
    global ques
    global answers
    c = 1
    for i in data:
        for j in data[i]:
            ques[str(c) + "__" + str(i)] = "What is the meaning of " + j[0] + " in " + "\"" + j[1].replace("\n",
                                                                                                           " ") + "\"" \
                                           + "?"
            answers[str(c) + "__" + str(i)] = data[i][j]
            c += 1
    global ans
    ans = {i: [answers[i]["correct"]] + answers[i]["wrong"][:3] for i in answers}
    global questions
    questions = {ques[i]: ans[i] for i in ques}
    global rev_index
    rev_index = {ques[i]: i for i in ques}
    questions_shuffled = shuffle(questions)
    for i in questions.keys():
        random.shuffle(questions[i])
    return render_template('quiz.html', q=questions_shuffled, o=questions)


@app.route('/quiz_result', methods=['POST', 'GET'])
def quiz_answers():
    global correct
    for i in list(questions.keys()):
        if i in request.form:
            answered = request.form[i]
        else:
            continue
        if answers[rev_index[i]]["correct"] == answered:
            if rev_index[i].split("__")[1] not in correct:
                correct[rev_index[i].split("__")[1]] = 0
            correct[rev_index[i].split("__")[1]] += 1
    return render_template('result.html', correct=sum(list(correct.values())))


@app.route('/modified', methods=['POST', 'GET'])
def final_text():
    final = q.final_text(correct, clusters)
    result = {"modified": final}
    return render_template('resource.html', success=result)


if __name__ == '__main__':
    app.run(debug=True)
