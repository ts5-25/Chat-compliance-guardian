import os
import logging

from flask import Flask, request, jsonify

import slack_sdk
from slack_sdk.errors import SlackApiError

import pickle
from openai import OpenAI

from sklearn.decomposition import PCA

with open('pca_model_v2.pkl', mode='rb') as file:
    pca = pickle.load(file)
with open('model_H_SVM_pca_v2.pkl', mode='rb') as f:
    model_H_SVM = pickle.load(f)
with open('model_L_SVM_pca_v2.pkl', mode='rb') as f:
    model_L_SVM = pickle.load(f)
with open('model_H_SVM_pca_v2.pkl', mode='rb') as f:
    model_logi = pickle.load(f)
    
# 環境変数の読み込み(local)
#bot_token = os.environ["SLACK_BOT_TOKEN"]
#slack_signing_secret = os.environ["SLACK_SIGNING_SECRET"]
#api_key = os.environ["OPENAI_API_KEY"]

# 環境変数の読み込み
bot_token = "SLACK_BOT_TOKEN"
slack_signing_secret = "SLACK_SIGNING_SECRET"
api_key = "OPENAI_API_KEY"

openai_client = OpenAI(
    api_key=api_key
)

def get_embedding_small(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

def AnalyzeMessage(message):
    a = []
    embedding = get_embedding_small(message)
    a.append(embedding)
    pca_vec = pca.transform(a)
    pred1 = model_logi.predict(pca_vec)
    pred2 = model_L_SVM.predict(pca_vec)
    pred3 = model_H_SVM.predict(pca_vec)
    if  (pred1==1) and (pred2==1) and (pred3==1):
        return True
    else:
        return False

# Flaskアプリケーションの設定
app = Flask(__name__)

client = slack_sdk.WebClient(token=bot_token)

#BOT_USER_ID = client.api_call("auth.test")['user_id']

admin_channel = "C06L61T11MK" #ハラスメント報告
message_channel = "C06LBP5BB0U" #メッセージ検知を報告
bot_id = ""

@app.route('/slack/events', methods=['POST'])
def respond_message():
    # Slackから送られてくるPOSTリクエストのBodyの内容を取得
    json = request.json
    print(json)

    # レスポンス用のJSONデータを作成
    # 受け取ったchallengeのKey/Valueをそのまま返却する
    if 'challenge' in json:
        challenge = json["challenge"]
    else:
        challenge = ""
        
    d = {'challenge' : challenge}
        # レスポンスとしてJSON化して返却
    try:
        if 'event' in json:
            event = json['event']
            if event["type"] == "message":
                if 'user' in event:
                    # 投稿のチャンネルID、ユーザーID、投稿内容を取得
                    channel_id = event['channel']
                    user_id = event['user']
                    text = event['text']
                    ts = event['ts']
                    if user_id != bot_id:
                        print("id:", user_id)    
                        client.chat_update(
                            channel=message_channel,
                            ts='1708664868.525089',
                            text="message検知"
                        )
                        if AnalyzeMessage(text):
                            print(text)
                            client.reactions_add(
                                channel=channel_id,
                                name="attention",
                                timestamp=ts
                            )
                            client.reactions_add(
                                channel=channel_id,
                                name="harassment",
                                timestamp=ts
                            )
                            response = client.chat_getPermalink(channel=channel_id, message_ts=ts)
                            link = response['permalink']
                            
                            client.chat_postMessage(channel=admin_channel, text=f"ハラスメントの疑いを検知しました:\n「{text}」\n{link}")
                            
    except SlackApiError as e:
        assert e.response["error"]
    return jsonify(d)

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.DEBUG)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
