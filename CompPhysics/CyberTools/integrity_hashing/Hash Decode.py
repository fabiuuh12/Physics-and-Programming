import hashlib

def crack_hash(hash_to_crack, wordlist_file, algorithm='md5'):
    try:
        with open(wordlist_file, 'r', encoding='utf-8') as file:
            for word in file:
                word = word.strip()
                if algorithm == 'md5':
                    hash_word = hashlib.md5(word.encode()).hexdigest()
                elif algorithm == 'sha1':
                    hash_word = hashlib.sha1(word.encode()).hexdigest()
                elif algorithm == 'sha256':
                    hash_word = hashlib.sha256(word.encode()).hexdigest()
                else:
                    print(f"Unsupported hash algorithm: {algorithm}")
                    return

                if hash_word == hash_to_crack:
                    print(f"[+] Password found: {word}")
                    return
        print("[-] Password not found in wordlist.")
    except FileNotFoundError:
        print("Wordlist file not found.")

if __name__ == "__main__":
    hash_input = input("Enter the hash to crack: ").strip()
    algo = input("Enter hash algorithm (md5/sha1/sha256): ").strip().lower()
    wordlist = input("Path to wordlist file (e.g., rockyou.txt): ").strip()

    crack_hash(hash_input, wordlist, algo)
