/**
 * @file 71A.CPP
 * @brief This program implements a word abbreviation logic.
 * Words longer than 10 characters are abbreviated to their first letter,
 * followed by the count of characters between the first and last letters,
 * and then the last letter. Words 10 characters or shorter remain unchanged.
 * This is a common string manipulation problem in competitive programming.
 *
 * Algorithm: String length check and formatting.
 * 1. Read the number of words to process.
 * 2. For each word:
 *    a. Read the word.
 *    b. Check its length.
 *    c. If the length is greater than 10, construct and print the abbreviated form.
 *    d. Otherwise, print the word as is.
 * Time Complexity: O(Sum of all word lengths) where each character is processed at most once.
 * Space Complexity: O(Max_Word_Length) to store the current word.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string> // Added for string manipulation

/**
 * @brief Main function of the program.
 * Reads words, applies the abbreviation logic if necessary, and prints the result.
 * @pre Input consists of an integer `n` (number of words), followed by `n` word strings.
 * @post For each word, prints its abbreviated form (if applicable) to standard output, each on a new line.
 * @return 0 upon successful execution.
 */
int main() {
  // Functional Utility: `n` stores the total number of words to process.
  int n, k; // `k` is declared but not used in this specific implementation, adhering to zero code mutation.
  std::string word; // Functional Utility: `word` stores the current word being processed.

  // Functional Utility: Reads the total number of words `n`.
  std::cin >> n;
  // std::cout << n; // Original commented line, preserved as per instruction to not alter code.

  // Block Logic: Iterates `n` times to process each word.
  // Invariant: At each iteration, a new word is read and processed according to the abbreviation rules.
  for (int i = 0; i < n; i++) {
    // Functional Utility: Reads the current word.
    std::cin >> word;

    // Functional Utility: Calculates the length of the current word.
    int wordLength = word.length();

    // Block Logic: Checks if the word's length exceeds 10 characters to determine if abbreviation is needed.
    // Pre-condition: `wordLength` contains the length of the current `word`.
    if (wordLength > 10) {
      // Functional Utility: Constructs the abbreviated form: first character + (length - 2) as string + last character.
      word =
          word.at(0) + std::to_string(wordLength - 2) + word.at(wordLength - 1);
    }

    // Functional Utility: Prints the processed word (abbreviated or original) followed by a newline.
    std::cout << word << std::endl;
  }

  return 0;
}
