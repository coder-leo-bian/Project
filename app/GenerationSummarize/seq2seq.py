from keras.models import Model
from keras.layers import LSTM, Embedding, Input, Dense

HIDDEN_UNITS = 128




class Seq2Seq:

    def __init__(self, config):
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.num_input_tokens = config['num_input_tokens']
        self.num_target_tokens = config['num_target_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.max_target_seq_length = config['max_target_seq_length']
        self.config = config
        self.encoder_model = None
        self.encoder_inputs = None
        self.decoder_model = None
        self.model = None
        self.encoder()
        self.decoder()

    def encoder(self):
        encoder_inputs = Input(shape=(None,), name='encoder_name')
        encoder_embedding = Embedding(input_dim=self.num_input_tokens,
                                      output_dim=self.max_target_seq_length,
                                      input_length=self.max_input_seq_length)(encoder_inputs)

        encoder_lstm = LSTM(HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding)
        encoder_states = [encoder_state_h, encoder_state_c]
        self.encoder_model = Model(encoder_inputs, encoder_states)
        self.encoder_inputs= encoder_inputs

    def decoder(self):
        # decoder输入
        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        # decoder 使用的循环神经网络
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True,
                            return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs)
        # 全连接层
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax',
                                name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([self.encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='categorical_crossentropy', loss='rmsprop')
        self.model = model
        self.decoder_model = Model(decoder_inputs, decoder_outputs)

        decoder_inputs_state = [Input(shape=(None, HIDDEN_UNITS)), Input(shape=(None, HIDDEN_UNITS))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_inputs_state)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_inputs_state, [decoder_outputs] + decoder_states)

    def fit(self):
        pass