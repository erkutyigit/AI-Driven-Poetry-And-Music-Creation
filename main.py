import numpy as np
import re
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from music21 import stream, note
import random


def clean_text(text):
    text = re.sub(r',', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'\"', '', text)
    text = re.sub(r'\(', '', text)
    text = re.sub(r'\)', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'“', '', text)
    text = re.sub(r'”', '', text)
    text = re.sub(r'’', '', text)
    text = re.sub(r'\.', '', text)
    text = re.sub(r';', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'\-', '', text)
    return text

with open('poems.txt', encoding='utf-8') as story:
    story_data = story.read()

lower_data = story_data.lower()
split_data = lower_data.splitlines()
final = ''
for line in split_data:
    line = clean_text(line)
    final += '\n' + line

final_data = final.split('\n')

max_vocab = 10000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(final_data)

word2idx = tokenizer.word_index
print("Number of unique words:", len(word2idx))
print("Word indices:", word2idx)
vocab_size = len(word2idx) + 1

input_seq = []
for line in final_data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_seq.append(n_gram_seq)

print("Length of input_seq:", len(input_seq))

max_seq_length = max(len(x) for x in input_seq)
print("Max sequence length:", max_seq_length)

input_seq = np.array(pad_sequences(input_seq, maxlen=max_seq_length, padding='pre'))
xs = input_seq[:, :-1]
labels = input_seq[:, -1]

np.save('data.npy', labels)

ys = to_categorical(labels, num_classes=vocab_size)

i = Input(shape=(max_seq_length - 1, ))
x = Embedding(vocab_size, 124)(i)
x = Dropout(0.2)(x)
x = LSTM(120, return_sequences=True)(x)
x = Bidirectional(layer=LSTM(150, return_sequences=True))(x)
x = GlobalMaxPooling1D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(vocab_size, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

r = model.fit(xs, ys, epochs=80)

def predict_poem(seed, no_lines, words_per_line):
    for i in range(no_lines):
        poem_line = seed
        for j in range(words_per_line):
            token_list = tokenizer.texts_to_sequences([seed])[0]
            token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
            predicted = np.argmax(model.predict(token_list), axis=1)

            new_word = ''
            for word, index in tokenizer.word_index.items():
                if predicted == index:
                    new_word = word
                    break
            seed += " " + new_word
            poem_line += " " + new_word

        print(poem_line)
        seed = seed.split()[np.random.randint(0, len(seed.split()))]  # Yeni seed'ı rastgele seç

seed_text = 'Freezy'
no_lines = 7
words_per_line = 10
predict_poem(seed_text, no_lines, words_per_line)


class MinorMusicGenerator:
    # notes in music21 are enumerated from 0 to 127
    # C4 (middle C) = 60
    # scale is a number from 59 (B) to 70 (Bb)
    def __init__(self, scale=60):
        self.minor_chords = None
        self.correct_notes = None
        self.baselines = None
        self.additional_chords = None
        # check if scale is integer and in a range of (59, 70)
        if not isinstance(scale, int):
            raise ValueError("scale must be an integer")
        elif scale < 59 or scale > 70:
            raise ValueError("scale must be in a range from 59 to 70")
        else:
            # If the scale is valid, it sets the scale attribute, and then calls two methods:
            self.scale = scale
        self.correct_minor_chords()
        self.create_baselines()
        self.calculate_correct_notes()
        self.add_additional_chords()

    # calculates a list of corrected notes based on a predefined set of shifts.
    # store the result in the correct_notes attribute.
    def calculate_correct_notes(self):
        shifts = [0, 2, 3, 5, 7, 8, 10]
        notes = [(self.scale + shift) for shift in shifts]
        self.correct_notes = notes

    # creates a minor chord based on a given note
    @classmethod
    def get_minor_chord(cls, note):
        return [note, note + 3, note + 7]

    # creates three minor chords using the get_minor_chord method.
    # The chords are based on the current scale, shifted by specific values.
    # The resulting chords are stored in the minor_chords attribute.
    def correct_minor_chords(self):
        first_chord = self.get_minor_chord(self.scale - 12)
        second_chord = self.get_minor_chord(self.scale + 3 - 12)
        third_chord = self.get_minor_chord(self.scale + 5 - 12)
        self.minor_chords = [first_chord, second_chord, third_chord]

    # creates additional chords
    # The resulting chords are stored in the additional_chords attribute.
    def add_additional_chords(self):
        chord1 = [self.scale, self.scale + 4, self.scale + 2, self.scale + 9]
        chord2 = [self.scale - 3, self.scale + 2, self.scale + 5, self.scale + 7]
        chord3 = [self.scale + 5, self.scale + 5, self.scale + 8, self.scale + 12]
        chord4 = [self.scale + 4, self.scale + 7, self.scale + 8]
        chord5 = [self.scale, self.scale + 3, self.scale + 5]
        self.additional_chords = [chord1, chord2, chord3, chord4, chord5]

    # creates a sequence of notes for the left hand (12 notes)
    @staticmethod
    def create_one_baseline(scale):
        cur_note = scale - 24
        return [cur_note, cur_note + 3, cur_note + 7, cur_note + 12,
                cur_note + 15, cur_note + 19, cur_note + 24, cur_note + 19,
                cur_note + 15, cur_note + 12, cur_note + 7, cur_note + 3]

    # creates 3 different sequences of notes for the left hand (from I, IV, V)
    def create_baselines(self):
        first_baseline = self.create_one_baseline(self.scale)
        second_baseline = self.create_one_baseline(self.scale + 3)
        third_baseline = self.create_one_baseline(self.scale + 5)
        self.baselines = [first_baseline, second_baseline, third_baseline]


def generate_music(scale: int, filepath: str, used_words: list):
    OCTAVE_SHIFT = 12
    new_song_generator = MinorMusicGenerator(scale)
    myStream = stream.Stream()

    intervals = 30
    note_duration = [4, 2, 1, 0.66]
    number_of_notes = [2, 2, 8, 12]

    volumes = [100, 50, 60, 60, 70, 80, 100, 80, 70, 60, 50, 50]

    def add_one_interval(current_index=0, right_hand_shift: int = 0,
                         current_velocity: int = 90, left_hand_shift: int = 0):
        current_index_for_the_right_hand = current_index
        current_note_duration_index = random.randint(0, len(note_duration) - 1)
        current_number_of_notes = number_of_notes[current_note_duration_index]
        current_duration = note_duration[current_note_duration_index]
        shift: int = right_hand_shift * OCTAVE_SHIFT

        for note_i in range(current_number_of_notes):
            if random.randint(0, 8) % 7 != 0:
                random_note = new_song_generator.correct_notes[random.randint(0, 6)] + shift
                my_note = note.Note(random_note, quarterLength=current_duration + 1)
                my_note.volume.velocity = current_velocity
                myStream.insert(current_index_for_the_right_hand, my_note)
            current_index_for_the_right_hand += current_duration

        sequence_of_notes = new_song_generator.baselines[random.randint(0, 2)]

        for note_i in range(0, 12):
            cur_note = sequence_of_notes[note_i]
            if random.randint(0, 8) % 7 != 0:
                for note_in_a_chord in range(len(sequence_of_notes)):
                    note_in_a_chord += OCTAVE_SHIFT * left_hand_shift
                new_note = note.Note(cur_note, quarterLength=1)
                new_note.volume.velocity = volumes[note_i]
                myStream.insert(current_index, new_note)
            current_index += 0.33

    for i in range(intervals):
        add_one_interval(current_index=4 * i,
                         right_hand_shift=random.randint(-1, 1),
                         current_velocity=random.randint(80, 110),
                         left_hand_shift=random.randint(-3, -1))
    add_one_interval(current_index=4 * intervals, current_velocity=50)
    myStream.write('midi', fp=filepath)

scale_for_music = 64
music_filepath = 'output_music.mid'
generate_music(scale_for_music, music_filepath, [])


