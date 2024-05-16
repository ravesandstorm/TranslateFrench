import torch
import random
import re
from Train import EncoderRNN, AttnDecoderRNN, MAX_LENGTH, device, tensorFromSentence, prepareData, SOS_token, EOS_token, hidden_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loadEncoderModel(input_size, hidden_size, model_path):
    encoder = EncoderRNN(input_size, hidden_size)
    if torch.cuda.is_available():
        encoder.load_state_dict(torch.load(model_path))
    else:
        encoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return encoder

def loadDecoderModel(hidden_size, output_size, model_path):
    decoder = AttnDecoderRNN(hidden_size, output_size)
    if torch.cuda.is_available():
        decoder.load_state_dict(torch.load(model_path))
    else:
        decoder.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return decoder

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, cont=True):
    while cont:
        print('1. n Random')
        print('2. Input Test Case')
        print('X. Exit')
        ch = int(input())
        if ch==1:
            print("Enter n: ", end='')
            n = int(input())
            for _ in range(n):
                pair = random.choice(pairs)
                print('>', pair[0])
                print('=', pair[1])
                output_words, attentions = evaluate(encoder, decoder, pair[0])
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
                print('')
        elif ch==2:
            while True:
                print('Input formatted french: (Exit to end)')
                sentence = input()
                if sentence.lower == 'exit':
                    break
                output_words, attentions = evaluate(encoder, decoder, sentence)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
                print('')
        else:
            cont=False

if __name__ == "__main__":
    # Load preprocessed data
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    
    # Define paths to saved encoder and decoder models
    encoder_model_path = 'encoderweights.pth'
    decoder_model_path = 'decoderweights.pth'
    
    # Load encoder and decoder models
    encoder = loadEncoderModel(input_lang.n_words, hidden_size, encoder_model_path).to(device)
    decoder = loadDecoderModel(hidden_size, output_lang.n_words, decoder_model_path).to(device)
    
    # Set model to evaluation mode
    encoder.eval()
    decoder.eval()
    
    # Evaluate randomly generated input sentences
    evaluateRandomly(encoder, decoder)