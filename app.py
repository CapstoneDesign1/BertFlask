from flask import Flask, request, render_template
from KoBERT.kobert.pytorch_kobert import get_kobert_model
from KoBERT.kobert_hf.kobert_tokenizer import KoBERTTokenizer

import numpy as np
import torch
import os
import requests
import json
import torch.nn as nn
from bert_sentiment_analysis import bertmodel, BERTDataset,tok,vocab, max_len,batch_size

device = 'cpu'
# model.to(torch.device('cuda'))
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=6,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def predict(self, input_sentence):

        data = [input_sentence, '0']
        dataset_predict = [data]
        # print("1")
        predict_test = BERTDataset(dataset_predict, 0, 1, tok, vocab, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(predict_test, batch_size=batch_size, num_workers=5)
        # print("2")
        model.eval()
        # print("3")
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids)
            # print("model output: ")
            # print(out)
            test_eval = []
            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()

                if np.argmax(logits) == 0:
                    test_eval.append("기쁨이")
                elif np.argmax(logits) == 1:
                    test_eval.append("불안이")
                elif np.argmax(logits) == 2:
                    test_eval.append("당황이")
                elif np.argmax(logits) == 3:
                    test_eval.append("슬픔이")
                elif np.argmax(logits) == 4:
                    test_eval.append("분노가")
                elif np.argmax(logits) == 5:
                    test_eval.append("상처가")
            # return out
            return test_eval[0]
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
@app.route('/<string:input_sentence>')
def input_test(input_sentence=None):  # put application's code here
    return render_template('main.html', input_sentence=input_sentence)


@app.route('/predict', methods=['POST'])
def make_prediction():
    # params = json.loads(request.get_data())
    # params_str = ''
    # for key in params.key:
    #     params_str += 'key: {}, value: {}<br>'.format(key, params[key])
    # return params_str

    temp = request.form['input_sentence']
    print("입력한 문장 : " + temp)
    # data = [input_sentence, '0']
    # dataset_predict = [data]

    # model = BERTClassifier(bertmodel, 0.5).to(device)
    # model.load_state_dict(torch.load('/Users/dugunhee/bert/drivemodel.pt'), strict=False)
    model.load_state_dict(torch.load('/Users/Desktop/BertFlask/model_state_dict.pt', map_location=torch.device('cpu')))
    # model.eval()
    out = model.predict(temp)
    print("분석 결과 : " + out)
    return ">> 입력하신 내용에서 " + out + " 느껴집니다."
    # return ">> 입력하신 내용에서 " + temp + " 느껴집니다."

if __name__ == '__main__':
    app.run()
