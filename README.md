# AI-Driven-Poetry-And-Music-Creation

## Overview

AI-Driven-Poetry-And-Music-Creation is an innovative project that explores the intersection of artificial intelligence and the arts. It utilizes deep learning models to generate original poetry and music compositions. The project integrates natural language processing (NLP) techniques with recurrent neural networks (RNNs) for poetry generation, and leverages music theory principles combined with AI-driven logic for music creation.

## Features

Poetry Generation: The project uses a sequence-based neural network to generate lines of poetry. Starting with a seed word or phrase, the model predicts subsequent words to craft a full poem.

Music Composition: A music generation module creates minor chord progressions and sequences of notes based on user-defined musical scales. The output is structured as a MIDI file.

Customizable Inputs: Users can adjust various parameters, such as the initial seed text for poetry and the musical scale for compositions.

## Objectives
The main objective of this project is to demonstrate how AI can be applied creatively to produce artistic outputs, blending the domains of poetry and music. This project serves as a tool for exploring new ways of art generation using modern machine learning algorithms, providing a platform for creative experimentation and further research.

## Project Components

### 1. Poetry Generation

The poetry generation model is built using an LSTM (Long Short-Term Memory) network with the following features:

Text Preprocessing: The input text is cleaned and tokenized to prepare it for model training.

Model Architecture: The model uses multiple layers, including embedding, LSTM, and bidirectional layers to predict the next word in a sequence based on the previous context.

Training: The model is trained on a corpus of poems to learn meaningful patterns and structures in poetic language.

Generation: By providing a seed word and defining the number of lines and words per line, the model can generate poetry dynamically.

### 2. Music Creation

The music generation component produces minor chord progressions and melodies, based on:

Scale and Chord Progressions: The user sets a musical scale, and the system generates accompanying chord progressions.
Melodic Sequences: Using AI-generated patterns, the music engine composes melodies and harmonizes them with basslines and additional chords.
MIDI Output: The composed music is exported in MIDI format for further use or playback.

## Usage

### Poetry Generation
To generate a poem, simply define the seed_text, number of lines, and words per line using the predict_poem() function.

### Music Composition
Set the musical scale and file path for the output, then call the generate_music() function to create an original composition.

## Future Enhancements
Integration of different poetry styles and structures.
Support for additional musical scales and more complex chord progressions.
Expanding the project to include voice synthesis for performing the poetry alongside the music.
