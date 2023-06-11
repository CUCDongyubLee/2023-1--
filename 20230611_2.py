
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein # 레벤슈타인 라이브러리 호출

class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        best_match_index = None  # 가장 유사한 질문의 인덱스를 저장할 변수
        best_similarity = None  # 가장 높은 유사도를 저장할 변수

        for i, question in enumerate(self.questions):
            distance = Levenshtein.distance(input_sentence, question)  # 입력 문장과 질문 간의 레벤슈타인 거리 계산
            similarity = 1 - (distance / max(len(input_sentence), len(question)))  # 유사도 계산 (1에서 거리를 나눔)

            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i  # 현재 질문이 가장 유사할 경우 인덱스를 업데이트

        if best_match_index is not None:
            return self.answers[best_match_index]  # 가장 유사한 질문에 해당하는 답변 반환
        else:
            return None  # 가장 유사한 질문이 없을 경우 None 반환

# CSV 파일 경로를 지정하세요.
filepath = 'ChatbotData.csv'

# 간단한 챗봇 인스턴스를 생성합니다.
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)