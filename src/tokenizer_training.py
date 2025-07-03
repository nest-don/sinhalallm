import sentencepiece as spm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from transformers import AutoTokenizer
import os

class SinhalaTokenizerTrainer:
    """Train custom tokenizer for Sinhala text"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
    def train_sentencepiece_tokenizer(self, corpus_path: str, model_prefix: str = "sinhala_sp"):
        """Train SentencePiece tokenizer"""
        print(f"Training SentencePiece tokenizer with vocab size {self.vocab_size}")
        
        # SentencePiece training arguments
        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,  # High coverage for Sinhala
            model_type='bpe',  # Byte Pair Encoding
            pad_id=0,
            eos_id=1,
            unk_id=2,
            bos_id=3,
            user_defined_symbols=['<|user|>', '<|assistant|>', '<|end|>'],
            input_sentence_size=10000000,
            shuffle_input_sentence=True
        )
        
        print(f"SentencePiece model saved as {model_prefix}.model")
        return f"{model_prefix}.model"
    
    def train_huggingface_tokenizer(self, corpus_path: str, output_dir: str = "sinhala_tokenizer"):
        """Train HuggingFace tokenizer"""
        print(f"Training HuggingFace BPE tokenizer with vocab size {self.vocab_size}")
        
        # Initialize a tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # Customization
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        
        # Trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<eos>", "<unk>", "<bos>", "<|user|>", "<|assistant|>", "<|end|>"],
            initial_alphabet=self.get_sinhala_alphabet()
        )
        
        # Train on corpus
        tokenizer.train([corpus_path], trainer)
        
        # Post-processing
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        
        # Save tokenizer
        os.makedirs(output_dir, exist_ok=True)
        tokenizer.save(f"{output_dir}/tokenizer.json")
        
        # Create tokenizer config
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "auto_map": {
                "AutoTokenizer": ["tokenizer.json", None]
            },
            "model_max_length": 2048,
            "padding_side": "left",
            "truncation_side": "right",
            "special_tokens": {
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
                "pad_token": "<pad>"
            }
        }
        
        import json
        with open(f"{output_dir}/tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        
        print(f"HuggingFace tokenizer saved to {output_dir}")
        return output_dir
    
    def get_sinhala_alphabet(self):
        """Get Sinhala alphabet for tokenizer initialization"""
        # Sinhala Unicode range: 0D80–0DFF
        sinhala_chars = []
        
        # Sinhala letters
        for i in range(0x0D80, 0x0E00):
            sinhala_chars.append(chr(i))
        
        # Common punctuation and symbols
        common_chars = [' ', '.', ',', '!', '?', ':', ';', '"', "'", '-', '(', ')', '[', ']', '{', '}']
        
        # Numbers
        numbers = [str(i) for i in range(10)]
        
        return sinhala_chars + common_chars + numbers
    
    def test_tokenizer(self, tokenizer_path: str, test_texts: list):
        """Test the trained tokenizer"""
        print(f"\nTesting tokenizer from {tokenizer_path}")
        
        if tokenizer_path.endswith('.model'):
            # SentencePiece tokenizer
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            
            for text in test_texts:
                tokens = sp.encode(text, out_type=str)
                decoded = sp.decode(tokens)
                print(f"Original: {text}")
                print(f"Tokens: {tokens}")
                print(f"Decoded: {decoded}")
                print(f"Token count: {len(tokens)}")
                print("-" * 50)
        
        else:
            # HuggingFace tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                
                for text in test_texts:
                    tokens = tokenizer.tokenize(text)
                    token_ids = tokenizer.encode(text)
                    decoded = tokenizer.decode(token_ids)
                    print(f"Original: {text}")
                    print(f"Tokens: {tokens}")
                    print(f"Token IDs: {token_ids}")
                    print(f"Decoded: {decoded}")
                    print(f"Token count: {len(tokens)}")
                    print("-" * 50)
            except Exception as e:
                print(f"Error testing tokenizer: {e}")
    
    def create_vocab_analysis(self, tokenizer_path: str):
        """Analyze vocabulary composition"""
        print(f"\nAnalyzing vocabulary from {tokenizer_path}")
        
        if tokenizer_path.endswith('.model'):
            sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
            vocab_size = sp.vocab_size()
            
            print(f"Vocabulary size: {vocab_size}")
            
            # Sample some tokens
            print("Sample vocabulary:")
            for i in range(min(50, vocab_size)):
                piece = sp.id_to_piece(i)
                print(f"ID {i}: '{piece}'")
        
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                vocab = tokenizer.get_vocab()
                print(f"Vocabulary size: {len(vocab)}")
                
                print("Sample vocabulary:")
                for i, (token, id_) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:50]):
                    print(f"ID {id_}: '{token}'")
            except Exception as e:
                print(f"Error analyzing tokenizer: {e}")

def main():
    # Test texts in Sinhala
    test_texts = [
        "ශ්‍රී ලංකාව දකුණු ආසියාවේ පිහිටි දූපත් රටකි.",
        "කොළඹ ශ්‍රී ලංකාවේ වාණිජ අගනුවරයි.",
        "සිංහල භාෂාව ශ්‍රී ලංකාවේ ප්‍රධාන භාෂාවකි.",
        "පහත ප්‍රශ්නයට සිංහලෙන් පිළිතුරු දෙන්න:",
        "බුද්ධ ධර්මය ශ්‍රී ලංකාවේ ප්‍රධාන ආගමයි."
    ]
    
    trainer = SinhalaTokenizerTrainer(vocab_size=32000)
    
    # Check if corpus exists
    corpus_path = "sinhala_corpus.txt"
    if not os.path.exists(corpus_path):
        print(f"Corpus file {corpus_path} not found. Please run data_preparation.py first.")
        return
    
    # Train SentencePiece tokenizer
    sp_model = trainer.train_sentencepiece_tokenizer(corpus_path)
    
    # Train HuggingFace tokenizer
    hf_tokenizer_dir = trainer.train_huggingface_tokenizer(corpus_path)
    
    # Test both tokenizers
    print("\n" + "="*60)
    print("TESTING SENTENCEPIECE TOKENIZER")
    print("="*60)
    trainer.test_tokenizer(sp_model, test_texts)
    
    print("\n" + "="*60)
    print("TESTING HUGGINGFACE TOKENIZER")
    print("="*60)
    trainer.test_tokenizer(hf_tokenizer_dir, test_texts)
    
    # Vocabulary analysis
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    trainer.create_vocab_analysis(sp_model)
    trainer.create_vocab_analysis(hf_tokenizer_dir)

if __name__ == "__main__":
    main() 