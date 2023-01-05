import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, pipeline, Wav2Vec2CTCTokenizer, \
    Wav2Vec2FeatureExtractor
import time
import scipy.signal as sps

# Improvements:
# - gpu / cpu flag
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2Inference():
    def __init__(self,model_name, hotwords=[], use_lm_if_possible = True):
        # if use_lm_if_possible:
        #     self.processor = AutoProcessor.from_pretrained(model_name)
        # else:
        #     tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]",
        #                                      word_delimiter_token="|")
        #     feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
        #                                                  do_normalize=True,
        #                                                  return_attention_mask=True)
        #     self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        self.tokenizer = Wav2Vec2CTCTokenizer("./vocab_20p20g20b.json", unk_token="[UNK]", pad_token="[PAD]",
                                         word_delimiter_token="|")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=True)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible

    def buffer_to_text(self,audio_buffer):
        if(len(audio_buffer)==0):
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values, attention_mask=inputs.attention_mask).logits            

        if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
            transcription = \
                self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      #hotword_weight=self.hotword_weight,  
                                      output_word_offsets=True,                                      
                                   )                             
            confidence = transcription.lm_score / len(transcription.text.split(" "))
            transcription = transcription.text       
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits,predicted_ids)

        return transcription, confidence   

    def confidence_score(self,logits,predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self,filename):
        audio_input, samplerate = sf.read(filename)
        #audio_input = sf.resample(audio_input, samplerate, 16000)
        if samplerate != 16000:
            number_of_samples = round(len(audio_input) * float(16000) / samplerate)
            audio_input = sps.resample(audio_input, number_of_samples)

        start = time.perf_counter()
        text, confidence = self.buffer_to_text(audio_input)
        inference_time = time.perf_counter() - start
        sample_length = len(audio_input) / 16000
        return (text, sample_length, inference_time, confidence)
    
if __name__ == "__main__":
    print("Model test")
    asr = Wave2Vec2Inference("oliverguhr/wav2vec2-large-xlsr-53-german-cv9")
    text = asr.file_to_text("test.wav")

    print(text)