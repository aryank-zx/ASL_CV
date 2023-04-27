from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = []
for i in range(0,20):
    colors.append((147, 98, 33))
print(len(colors))
def prob_viz(res, actions, input_frame, colors,threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# 1. New detection variables
sequence = []
sentence = []
accuracy=[]
predictions = []
threshold = 0.8 

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("https://192.168.43.41:8080/video")
# Set mediapipe model 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        cropframe=frame[40:400,0:300]
        # print(frame.shape)
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (147, 98, 33), 2)
       
        image, results = mediapipe_detection(cropframe, hands)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try: 
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)]*100)) 

                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy=accuracy[-1:]
        except Exception as e:
            # print(e)
            pass
                # Define font sizes
        cv2.rectangle(frame, (0,0), (300, 40), (147, 98, 33), -1)
        font_size1 = 0.7  
        font_size2 = 0.6  

# Define colors
        black = (0, 0, 0)
        white = (255, 255, 255)
        light_red = (85, 49, 245)

        accuracy_formatted = ['{:.4f}'.format(float(acc)) for acc in accuracy]
        cv2.putText(frame, "Output:", (3, 30), cv2.FONT_HERSHEY_DUPLEX, font_size1, black, 1, cv2.LINE_AA)
        cv2.putText(frame, ' '.join(sentence), (95, 30), cv2.FONT_HERSHEY_DUPLEX, font_size1, white, 1, cv2.LINE_AA)
        cv2.putText(frame, ' '.join(accuracy_formatted), (120, 30), cv2.FONT_HERSHEY_DUPLEX, font_size2, light_red, 1, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()