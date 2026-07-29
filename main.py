import sys
from colorama import Fore, Style, init
from src.detector import EmailSpamDetector

init(autoreset=True)


def main():
    print(Fore.CYAN + Style.BRIGHT + """
==================================================   
    Python Spam Detector Pipeline 
==================================================
        """)

    detector = EmailSpamDetector()

    if not detector.load_dataset():
        sys.exit(1)

    print(f"\n{Fore.BLUE}[*] {Style.RESET_ALL}Training Model...")
    detector.train(split_ratio=0.8)

    print(f"{Fore.BLUE}[*] {Style.RESET_ALL}Evaluating model accuracy on test data...")
    detector.evaluate_model()

    print(f"{Style.DIM}--- (Type 'exit' or 'q' to quit) ---")
    while True:
        try:
            user_input = input(
                f"\n{Fore.YELLOW}Enter message text to analyze > {Style.RESET_ALL}"
            ).strip()

            if not user_input:
                continue
            if user_input.lower() in ["exit", "q", "quit"]:
                print(f"{Fore.MAGENTA}Exiting detector. bye!")
                break

            result = detector.predict(user_input)

            is_spam = "SPAM" in result["label"]
            status_color = (
                Fore.RED + Style.BRIGHT if is_spam else Fore.GREEN + Style.BRIGHT
            )

            print(f"\n{Fore.CYAN}--- Detection Result ---")
            print(f"Classification : {status_color}{result['label']}")
            print(f"Confidence     : {Fore.WHITE}{Style.BRIGHT}{result['confidence']}%")
            print(f"Spam Score : {Fore.RED}{result['spam_score']}")
            print(f"Ham Score  : {Fore.GREEN}{result['ham_score']}")
            print(f"{Fore.CYAN}------------------------")

        except KeyboardInterrupt:
            print(f"\n{Fore.MAGENTA}Exiting...")
            break


if __name__ == "__main__":
    main()
