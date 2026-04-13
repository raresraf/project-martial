/**
 * @file 71A.CPP
 * @brief This program implements the "Way Too Long Words" problem.
 * It reads 'n' words and abbreviates those longer than 10 characters.
 * Abbreviation: First character + (length - 2) as a string + last character.
 * Algorithm: Dynamic array for words, string length check, and conditional abbreviation using `to_string`.
 * Time Complexity: O(N * L), where N is the number of words and L is the average length of a word.
 * Space Complexity: O(N * L) for storing all words in a dynamic array.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <string.h> // C-style string header, not strictly needed for std::string but often included.
#include <string>   // Required for std::string and std::to_string.

using namespace std;

int main() {
  int n; // Stores the number of words to process.
  // Reads the number of words that will be entered.
  cin >> n;
  // Dynamically allocates an array of 'n' std::string objects to store the words.
  string *words = new string[n];
  /**
   * Block Logic: Reads 'n' words from standard input and stores them in the dynamically allocated array.
   * Invariant: After this loop, the 'words' array contains all 'n' input words.
   */
  for (int i = 0; i < n; i++) {
    cin >> words[i];
  }
  /**
   * Block Logic: Iterates through the array of words, applying the abbreviation rule to each.
   * Words with length 10 or less are printed as is. Longer words are abbreviated.
   */
  for (int i = 0; i < n; i++) {
    if (words[i].length() <= 10)
      // Prints the original word if its length is 10 or less.
      cout << words[i] << endl;
    else {
      // Extracts the first character of the word.
      string firstL(1, words[i].at(0));
      // Calculates the number of characters between the first and last (length - 2)
      // and converts it to a string.
      string middleN = to_string((1, words[i].length() - 2)); // The (1, ...) is a common competitive programming idiom for sequence point.
      // Extracts the last character of the word.
      string lastL(1, words[i].at(words[i].length() - 1));
      // Concatenates and prints the abbreviated word.
      cout << firstL + middleN + lastL << endl;
    }
  }
  // Deallocates the dynamically allocated array to prevent memory leaks.
  delete[] words;
  // Standard program exit.
  return 0;
}
