/**
 * @file 118A.CPP
 * @brief Solution for the "String Task" problem from Codeforces.
 *
 * @details
 * This program transforms a string by deleting all vowels, inserting a '.'
 * before each consonant, and converting all consonants to lowercase. This
 * implementation uses a single-pass approach, iterating through the input
 * string and printing the transformed output directly to the console without
 * building an intermediate result string.
 *
 * @section
 * Algorithm: Single-pass direct-printing transformation.
 * Time Complexity: O(N), where N is the length of the input string.
 * Space Complexity: O(N) to store the input string.
 */
#include <ctype.h>
#include <iostream>
#include <string.h> // C-style string header, not required for std::string.
using namespace std;

int main() {
  string test;
  cin >> test;
  /**
   * @brief Main loop to process the string and print the output.
   *
   * Logic: The loop iterates character by character. Vowels are skipped.
   * Consonants are processed and printed immediately in the required format.
   */
  for (int i = 0; i < test.length(); i++) {

    // Pre-condition: Check if the character is a vowel (case-insensitive).
    if (test[i] == 'A' || test[i] == 'O' || test[i] == 'Y' || test[i] == 'E' ||
        test[i] == 'U' || test[i] == 'I' || test[i] == 'a' || test[i] == 'o' ||
        test[i] == 'y' || test[i] == 'e' || test[i] == 'u' || test[i] == 'i') {
      // If it's a vowel, skip to the next character.
      continue;
    } else {
      // Invariant: The character is a consonant.
      // The logic here handles both upper and lower case consonants.
      if (isupper(test[i])) {
        // If uppercase, convert it to lowercase in-place first.
        test[i] = tolower(test[i]);
        cout << "." << test[i];
      } else {
        // If already lowercase, just print in the required format.
        cout << "." << test[i];
      }
    }
  }
  return 0;
}
