/**
 * @6997c0ea-ba99-42c0-9779-5d98b2ff28a0/71A.CPP
 * @brief Problem: Way Too Long Words.
 *        If a word's length exceeds 10 characters, it's replaced by an abbreviation.
 *        The abbreviation consists of the first letter, the count of intermediate letters, and the last letter.
 *        Otherwise, the word remains unchanged.
 * Algorithm: String length checking and conditional abbreviation.
 *            For each word, check its length. If greater than 10, construct the abbreviation.
 *            Otherwise, print the original word.
 * Time Complexity: O(L_total) where L_total is the total number of characters across all words,
 *                  as each character is processed at most once.
 * Space Complexity: O(L_max) where L_max is the maximum length of a single word,
 *                   for storing the current word.
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string word;
  int num;
  // Block Logic: Reads the number of words to be processed from standard input.
  cin >> num;
  
  // Block Logic: Iterates 'num' times, processing one word in each iteration.
  // Pre-condition: 'num' is a positive integer representing the total count of words.
  // Invariant: In each iteration, a word is read and either abbreviated or printed as is.
  for (int i = 0; i < num; i++) {
    // Block Logic: Reads a single word from standard input.
    cin >> word;
    // Block Logic: Checks if the length of the current word exceeds 10 characters.
    // Pre-condition: 'word' contains the string to be processed.
    // Post-condition: If true, the word will be abbreviated; otherwise, printed as is.
    if (word.length() > 10) {
      // Functional Utility: Prints the abbreviated form of the word.
      // The abbreviation consists of the first character, the count of intermediate characters, and the last character.
      cout << word.at(0) << word.length() - 2 << word.at(word.length() - 1)
           << endl;
    } else {
      // Functional Utility: Prints the original word as its length is 10 or less.
      cout << word << endl;
    }
  }
  // The problem typically ends without an explicit return 0 in competitive programming,
  // but for completeness and good practice, it's often included.
  return 0; // Functional Utility: Indicates successful program execution.
}
