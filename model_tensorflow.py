import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import prepare_stopwords
import standard_data


def process(input, language):
    
    data = input
    # sns.countplot(x='spam', data=data)
    # plt.show()

    # Downsampling để cân bằng
    
    print("Downsampling")
    ham_msg = data[data.spam == 0]
    spam_msg = data[data.spam == 1]
    ham_msg = ham_msg.sample(n = len(spam_msg), random_state = 42) # lấy mẫu ngẫu nhiên, đặt trạng thái để đảm bảo nhất quán
    

    balanced_data = pd.concat([ham_msg, spam_msg], ignore_index = True)

    # sns.countplot(x='spam', data=balanced_data)
    # plt.show()

    # 2. Tiền xử lí dữ liệu
    print("Tiền xử lí dữ liệu")
    if language == "vi":
        balanced_data['text'] = balanced_data['text'].apply(lambda text: standard_data.standard_vi(text))
    else:
        balanced_data['text'] = balanced_data['text'].apply(lambda text: standard_data.standard_en(text))

    # 3. Chia dữ liệu
    print("Chia dữ liệu")
    # train_X chứa dữ liệu, train_Y chứa nhãn
    train_X, test_X, train_Y, test_Y = train_test_split(balanced_data['text'],
                                                        balanced_data['spam'],
                                                        test_size = 0.2,
                                                        random_state = 42)


    # 4. Tokenize dữ liệu
    print("Tokenize dữ liệu")
    tokenizer = Tokenizer()
    # xây dựng từ điển
    tokenizer.fit_on_texts(train_X)

    # chuyển thành chuỗi số
    train_sequences = tokenizer.texts_to_sequences(train_X)
    test_sequences = tokenizer.texts_to_sequences(test_X)

    # Đệm chuỗi -> cùng độ dài
    # thêm cuối bớt đầu
    max_len = 100 # độ dài tối đa
    train_sequences = pad_sequences(train_sequences, maxlen = max_len, padding = 'post', truncating = 'post')
    test_sequences = pad_sequences(test_sequences, maxlen = max_len, padding = 'post', truncating = 'post')


    # Mô hình
    # Điều chỉnh LSTM và thêm dropout để hiệu quả
    print("Mô hình")
    # tạo mô hình mạng nơ ron sử dụng gói Sequential
    model = tf.keras.models.Sequential()
    # thêm lớp nhúng
    # chuyển các số thành vector
    model.add(tf.keras.layers.Embedding(input_dim = len(tokenizer.word_index) + 1,
                                        output_dim = 32,
                                        input_length = max_len))
    # Long Short-Term Memory phục vụ xử lí chuỗi duy trì bộ nhớ ngắn hạn dài hạn 16 đơn vị
    model.add(tf.keras.layers.LSTM(16))
    # lớp kết nôi, tăng tính không tuyến tính max(0,x)
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    # lớp kết nôi, 0, 1
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    # dựa vào sigmoid có được kích hoạt không
    # đo bằng accuracy
    # adam tự điều chỉnh quá trình huấn luyện
    model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                metrics = ['accuracy'],
                optimizer = 'adam')

    # dừng sớm 3 vòng accuracy trở lại
    es = EarlyStopping(patience=3,
                    monitor = 'val_accuracy',
                    restore_best_weights = True)
    # 2 vòng loss ko cải thiện thì giảm đi 0.5
    lr = ReduceLROnPlateau(patience = 2,
                        monitor = 'val_loss',
                        factor = 0.5,
                        verbose = 0)


    # Train
    print("Train")
    # huấn luyện, đánh giá, 20 vòng, mỗi lần 32 mẫu
    history = model.fit(train_sequences, train_Y,
                        validation_data=(test_sequences, test_Y),
                        epochs=20,
                        batch_size=32,
                        callbacks = [lr, es]
                    )


    # Đánh giá
    print("Đánh giá")
    test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
    print('Test Loss :',test_loss)
    print('Test Accuracy :',test_accuracy)

    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend()
    # plt.show()

    return model, tokenizer