/**
 * @file 112A.CPP
 * @brief This program performs a case-insensitive lexicographical comparison of two input strings.
 * It converts both strings to lowercase and then compares them to determine their relative order.
 * Algorithm: Character-by-character case conversion followed by standard string comparison.
 * Time Complexity: O(L) where L is the maximum length of the input strings, due to character iteration.
 * Space Complexity: O(L) for storing the input strings.
 */
#include <iostream> // Required for input/output operations (cin, cout).
#include <stdio.h>  // C standard input/output library, usually not strictly needed with iostream for string.
#include <string.h> // C string manipulation library, not strictly needed with C++ string.
#include <cctype>   // Required for isupper.

using namespace std;
int main() {
  string word;  // Stores the first input string.
  string word2; // Stores the second input string.

  // Reads the two strings from standard input.
  cin >> word;
  cin >> word2;

  /**
   * Block Logic: Converts all uppercase letters in both 'word' and 'word2' to lowercase.
   * This ensures a case-insensitive comparison in the subsequent step.
   * Invariant: After this loop, both 'word' and 'word2' will contain only lowercase letters (and other original characters).
   */
  for (int s = 0; s < (int)word.length(); s++) {
    // Converts character in 'word' to lowercase if it's uppercase.
    if (isupper(word[s]))
      word[s] = word[s] + 32;
    // else; // No operation needed if already lowercase or not a letter.

    // Converts character in 'word2' to lowercase if it's uppercase.
    if (isupper(word2[s]))
      word2[s] = word2[s] + 32;
    // else; // No operation needed if already lowercase or not a letter.
  }

  /**
   * Conditional Logic: Compares the processed strings lexicographically.
   * Prints '0' if strings are equal.
   * Prints '-1' if 'word' is lexicographically smaller than 'word2'.
   * Prints '1' if 'word' is lexicographically greater than 'word2'.
   */
  if (word == word2)
    cout << "0" << endl;

  else if (word < word2)
    cout << "-1" << endl;

  else if (word > word2)
    cout << "1" << endl;

  // Standard program exit.
  return 0;
}
