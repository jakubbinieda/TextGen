import numpy as np
from collections import defaultdict

class LSTM:
    def __init__(self, hidden_size, max_words, sequences):
        self.hidden_size = hidden_size
        self.word_to_id, self.id_to_word, self.num_sentences, self.vocab_size = self.sequences_to_dicts(sequences, max_words)
        self.W_f, self.W_i, self.W_g, self.W_o, self.W_v, self.b_f, self.b_i, self.b_g, self.b_o, self.b_v = self.init_lstm(self.hidden_size, self.vocab_size, self.hidden_size + self.vocab_size)
        print(self.vocab_size)

    def sequences_to_dicts(self, sequences, max_words):
        flatten = lambda l: [item for sublist in l for item in sublist]        
        word_count = defaultdict(int)
        for word in flatten(sequences):
            word_count[word] += 1
        word_count = sorted(list(word_count.items()), key=lambda l: -l[1])
        unique_words = [item[0] for item in word_count]
        unique_words.append('INT')
        self.num_sentences, self.vocab_size = len(sequences), len(unique_words)
        self.word_to_id = defaultdict(lambda: max_words)
        self.id_to_word = defaultdict(lambda: 'INT')
        for id, word in enumerate(unique_words):
            self.word_to_id[word] = id
            self.id_to_word[id] = word
        return self.word_to_id, self.id_to_word, self.num_sentences, self.vocab_size

    def one_hot_encode(self, id, vocab_size):
        one_hot = np.zeros(vocab_size)
        one_hot[id] = 1.0
        return one_hot
    
    def one_hot_encode_sequence(self, sequence, vocab_size):
        encoding = np.array([self.one_hot_encode(self.word_to_id[word], vocab_size) for word in sequence])
        encoding = encoding.reshape(encoding.shape[0], encoding.shape[1], 1) 
        return encoding
    
    @staticmethod
    def sigmoid(x, der=False):
        x_safe = x + 1e-12
        f = 1 / (1 + np.exp(-x_safe))
        if der:
            return f * (1 - f)
        else:
            return f

    @staticmethod  
    def tanh(x, der=False):
        x_safe = x + 1e-12
        f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
        if der:
            return 1-f**2
        else:
            return f
 
    @staticmethod
    def softmax(x, der=False):
        x_safe = x + 1e-12
        f = np.exp(x_safe) / np.sum(np.exp(x_safe))
        if der:
            pass
        else:
            return f

    @staticmethod 
    def clip_grad(grads, max_norm=0.25):
        max_norm = float(max_norm)
        total = 0
        for g in grads:
            norm_sq = np.sum(np.square(g))
            total += norm_sq
        total = np.sqrt(total)
        coef = max_norm / (total + 1e-6)
        if coef < 1:
            for g in grads:
                g *= coef
        return grads

    def init_lstm(self, hidden_size, vocab_size, z_size):
        W_f = np.random.randn(hidden_size, z_size)
        b_f = np.zeros((hidden_size, 1))
        W_i = np.random.randn(hidden_size, z_size)
        b_i = np.zeros((hidden_size, 1))
        W_g = np.random.randn(hidden_size, z_size)
        b_g = np.zeros((hidden_size, 1))
        W_o = np.random.randn(hidden_size, z_size)
        b_o = np.zeros((hidden_size, 1))
        W_v = np.random.randn(vocab_size, hidden_size)
        b_v = np.zeros((vocab_size, 1))
        
        W_f = self.generate_orthogonal(W_f)
        W_i = self.generate_orthogonal(W_i)
        W_g = self.generate_orthogonal(W_g)
        W_o = self.generate_orthogonal(W_o)
        W_v = self.generate_orthogonal(W_v)

        return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v

    def generate_orthogonal(self, M):
        num_rows, num_cols = M.shape
        random_M = np.random.randn(num_rows, num_cols)
        if num_rows < num_cols:
            random_M = random_M.T
        q_M, r_M = np.linalg.qr(random_M)
        diag_r = np.diag(r_M, 0)
        sign_diag = np.sign(diag_r)
        q_M *= sign_diag
        if num_rows < num_cols:
            q_M = q_M.T
        res = q_M
        return res
    
    def forward(self, inputs, h_prev, C_prev):
        W_f, W_i, W_g, W_o, W_v = self.W_f, self.W_i, self.W_g, self.W_o, self.W_v
        b_f, b_i, b_g, b_o, b_v = self.b_f, self.b_i, self.b_g, self.b_o, self.b_v
        
        x_s, z_s, f_s, i_s,  = [], [] ,[], []
        g_s, C_s, o_s, h_s = [], [] ,[], []
        v_s, output_s =  [], [] 
        
        h_s.append(h_prev)
        C_s.append(C_prev)
        
        for x in inputs:
            z = np.row_stack((h_prev, x))
            z_s.append(z)
            
            f = self.sigmoid(np.dot(W_f, z) + b_f)
            f_s.append(f)
            
            i = self.sigmoid(np.dot(W_i, z) + b_i)
            i_s.append(i)
            
            g = self.tanh(np.dot(W_g, z) + b_g)
            g_s.append(g)
            
            C_prev = f * C_prev + i * g 
            C_s.append(C_prev)

            o = self.sigmoid(np.dot(W_o, z) + b_o)
            o_s.append(o)
            
            h_prev = o * self.tanh(C_prev)
            h_s.append(h_prev)

            v = np.dot(W_v, h_prev) + b_v
            v_s.append(v)
            
            output = self.softmax(v)
            output_s.append(output)

        return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s
    
    def backward(self, z, f, i, g, C, o, h, v, outputs, targets):
        W_f, W_i, W_g, W_o, W_v = self.W_f, self.W_i, self.W_g, self.W_o, self.W_v
        b_f, b_i, b_g, b_o, b_v = self.b_f, self.b_i, self.b_g, self.b_o, self.b_v

        W_f_d = np.zeros_like(W_f)
        b_f_d = np.zeros_like(b_f)

        W_i_d = np.zeros_like(W_i)
        b_i_d = np.zeros_like(b_i)

        W_g_d = np.zeros_like(W_g)
        b_g_d = np.zeros_like(b_g)

        W_o_d = np.zeros_like(W_o)
        b_o_d = np.zeros_like(b_o)

        W_v_d = np.zeros_like(W_v)
        b_v_d = np.zeros_like(b_v)

        dh_next = np.zeros_like(h[0])
        dC_next = np.zeros_like(C[0])

        for t in reversed(range(len(outputs))):
            dv = np.copy(outputs[t])
            dv[np.argmax(targets[t])] -= 1

            W_v_d += np.dot(dv, h[t].T)
            b_v_d += dv

            dh = np.dot(W_v.T, dv)
            dh += dh_next

            do = dh * self.tanh(C[t])
            do = self.sigmoid(o[t], der=True) * do

            W_o_d += np.dot(do, z[t].T)
            b_o_d += do

            dC = np.copy(dC_next)
            dC += dh * o[t] * self.tanh(C[t], der=True)

            dg = dC * i[t]
            dg = self.tanh(g[t], der=True) * dg

            W_g_d += np.dot(dg, z[t].T)
            b_g_d += dg

            di = dC * g[t]
            di = self.sigmoid(i[t], True) * di
            W_i_d += np.dot(di, z[t].T)
            b_i_d += di

            df = dC * C[t-1]
            df = self.sigmoid(f[t], True) * df
            W_f_d += np.dot(df, z[t].T)
            b_f_d += df

            dz = (np.dot(W_f.T, df) + np.dot(W_i.T, di) + np.dot(W_g.T, dg) + np.dot(W_o.T, do))

            dh_next = dz[:self.hidden_size, :]
            dC_next = f[t] * dC

        grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d
        grads = self.clip_grad(grads)
        return grads

    @staticmethod
    def cross_entropy_loss(predictions, targets):
        predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
        loss = -np.sum(targets * np.log(predictions)) / targets.shape[1]
        return loss

    def train(self, training_set, validation_set, num_epochs=50, lr=1e-1):
        training_loss = []
        validation_loss = []

        for epoch in range(num_epochs):
            epoch_training_loss = 0
            epoch_validation_loss = 0

            for inputs, targets in validation_set:
                inputs_one_hot = self.one_hot_encode_sequence(inputs, self.vocab_size)
                targets_one_hot = self.one_hot_encode_sequence(targets, self.vocab_size)
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))
                z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = self.forward(inputs_one_hot, h, c)
                loss = self.cross_entropy_loss(outputs, targets_one_hot)
                epoch_validation_loss += loss

            for inputs, targets in training_set:
                inputs_one_hot = self.one_hot_encode_sequence(inputs, self.vocab_size)
                targets_one_hot = self.one_hot_encode_sequence(targets, self.vocab_size)
                h = np.zeros((self.hidden_size, 1))
                c = np.zeros((self.hidden_size, 1))
                z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = self.forward(inputs_one_hot, h, c)
                loss = self.cross_entropy_loss(outputs, targets_one_hot)
                grads = self.backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot)
                
                params = [self.W_f, self.W_i, self.W_g, self.W_o, self.W_v, self.b_f, self.b_i, self.b_g, self.b_o, self.b_v]
                for idx, (param, grad) in enumerate(zip(params, grads)):
                    params[idx] = param - lr * grad
                (self.W_f, self.W_i, self.W_g, self.W_o, self.W_v, self.b_f, self.b_i, self.b_g, self.b_o, self.b_v) = params
                
                epoch_training_loss += loss

            training_loss.append(epoch_training_loss / len(training_set))
            validation_loss.append(epoch_validation_loss / len(validation_set))
            print(f'Epoch {epoch}, Cross-entropy train: {training_loss[-1]}, Cross-entropy val: {validation_loss[-1]}')

        return training_loss, validation_loss

    def generate_text(self, starting_sequence, length=25):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        generated_text = starting_sequence.copy()
        starting_sequence_encoded = self.one_hot_encode_sequence(starting_sequence, self.vocab_size)

        for _ in range(length):
            _, _, _, _, _, _, _, _, outputs = self.forward(starting_sequence_encoded, h, c)
            last_output = outputs[-1]
            next_word_id = np.random.choice(range(self.vocab_size), p=last_output.ravel())
            next_word = self.id_to_word[next_word_id]
            generated_text.append(next_word)
            starting_sequence_encoded = self.one_hot_encode_sequence([next_word], self.vocab_size)
        return generated_text
    
    def calculate_perplexity(self, sequences, max_words):
        total_loss = 0
        n_words = 0
        for sequence in sequences:
            input_words, target_words = sequence
            input_sequence = self.one_hot_encode_sequence(input_words, max_words)
            h_prev = np.zeros((self.hidden_size, 1))
            C_prev = np.zeros((self.hidden_size, 1))
            _, _, _, _, _, _, _, _, output_s = self.forward(input_sequence, h_prev, C_prev)
            for t, target in enumerate(target_words):
                total_loss += -np.log(output_s[t][self.word_to_id[target], 0])
                n_words += 1
        perplexity = np.exp(total_loss / n_words)
        return perplexity