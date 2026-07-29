import math
import re
from collections import Counter
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)


DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/siddhanthreddy2803/Spam-Detection-dataset/main/mail_data.csv"


class EmailSpamDetector:
    def __init__(self):
        self.spam_word_counts = Counter()
        self.ham_word_counts = Counter()
        self.total_spam_emails = 0
        self.total_ham_emails = 0
        self.vocab = set()
        self.df = None

    def load_dataset(self, file_path_or_url=DEFAULT_DATASET_URL):
        """Dataset loading via pandas"""
        try:
            print(f"{Fore.BLUE}[*] {Style.RESET_ALL}Fetching dataset using pandas...")
            # Load CSV
            self.df = pd.read_csv(file_path_or_url)

            self.df.columns = [col.strip().capitalize() for col in self.df.columns]

            if "Category" not in self.df.columns or "Message" not in self.df.columns:
                self.df.rename(
                    columns={
                        self.df.columns[0]: "Category",
                        self.df.columns[1]: "Message",
                    },
                    inplace=True,
                )

            print(
                f"{Fore.GREEN}{Style.BRIGHT}[✓] Loaded {len(self.df):,} rows from dataset!"
            )
            return True

        except Exception as e:
            print(
                f"{Fore.RED}{Style.BRIGHT}[!] Failed to load dataset: {Style.RESET_ALL}{e}"
            )
            return False

    def clean_text(self, text):
        """lowercased words"""
        if not isinstance(text, str):
            return []
        return re.findall(r"\b[a-zA-Z]+\b", text.lower())

    def train(self, split_ratio=0.8):
        if self.df is None or self.df.empty:
            print(f"{Fore.RED}[!] No data available to train.")
            return

        shuffled_df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_idx = int(len(shuffled_df) * split_ratio)

        train_df = shuffled_df.iloc[:split_idx]
        self.test_df = shuffled_df.iloc[split_idx:]

        # Reset
        self.spam_word_counts.clear()
        self.ham_word_counts.clear()
        self.total_spam_emails = 0
        self.total_ham_emails = 0
        self.vocab.clear()

        # training loop
        for _, row in train_df.iterrows():
            label = str(row["Category"]).strip().lower()
            text = str(row["Message"])
            words = self.clean_text(text)

            self.vocab.update(words)

            if label == "spam":
                self.total_spam_emails += 1
                self.spam_word_counts.update(words)
            elif label == "ham":
                self.total_ham_emails += 1
                self.ham_word_counts.update(words)

        print(
            f"{Fore.GREEN}[✓] Trained on {len(train_df):,} samples "
            f"({self.total_spam_emails:,} Spam | {self.total_ham_emails:,} Ham). "
            f"Vocabulary size: {len(self.vocab):,} unique words."
        )

    def evaluate_model(self):
        """Evaluates model performance"""
        if not hasattr(self, "test_df") or self.test_df.empty:
            print(f"{Fore.YELLOW}[!] No test dataset available for evaluation.")
            return

        correct = 0
        total = len(self.test_df)

        for _, row in self.test_df.iterrows():
            true_label = str(row["Category"]).strip().lower()
            text = str(row["Message"])

            prediction = self.predict(text, silent=True)
            pred_label = "spam" if "SPAM" in prediction["label"] else "ham"

            if pred_label == true_label:
                correct += 1

        accuracy = round((correct / total) * 100, 2)

        print(f"\n{Fore.CYAN}{Style.BRIGHT}==========================================")
        print(f"{Fore.CYAN}{Style.BRIGHT}        MODEL ACCURACY EVALUATION         ")
        print(f"{Fore.CYAN}{Style.BRIGHT}==========================================")
        print(f"Total Test Samples : {total:,}")
        print(f"Correct Predictions: {correct:,}")
        print(f"Accuracy Score     : {Fore.GREEN}{Style.BRIGHT}{accuracy}%")
        print(f"{Fore.CYAN}==========================================\n")

    def predict(self, email_text, silent=False):
        """Scores text against spam vs ham word distributions."""
        words = self.clean_text(email_text)
        total_emails = self.total_spam_emails + self.total_ham_emails

        if total_emails == 0:
            if not silent:
                print(f"{Fore.YELLOW}[!] Warning: Model is untrained.")
            return {
                "label": "UNKNOWN",
                "confidence": 0,
                "spam_score": 0,
                "ham_score": 0,
            }

        # Log probabilities
        spam_score = math.log(self.total_spam_emails / total_emails)
        ham_score = math.log(self.total_ham_emails / total_emails)

        total_spam_words = sum(self.spam_word_counts.values())
        total_ham_words = sum(self.ham_word_counts.values())
        vocab_size = len(self.vocab)

        # Log likelihood addition  (+1)
        for word in words:
            if word in self.vocab:
                p_spam = (self.spam_word_counts[word] + 1) / (
                    total_spam_words + vocab_size
                )
                p_ham = (self.ham_word_counts[word] + 1) / (
                    total_ham_words + vocab_size
                )

                spam_score += math.log(p_spam)
                ham_score += math.log(p_ham)

        is_spam = spam_score > ham_score
        score_diff = abs(spam_score - ham_score)
        confidence = round((1 / (1 + math.exp(-min(score_diff, 700)))) * 100, 2)

        return {
            "label": "🚨 SPAM" if is_spam else "✅ CLEAN (HAM)",
            "confidence": confidence,
            "spam_score": round(spam_score, 2),
            "ham_score": round(ham_score, 2),
        }
