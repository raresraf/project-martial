/**
 * @file 118A.CPP
 * @brief Solution to Codeforces Problem 118A: "String Task".
 *        This program takes a word as input, converts all uppercase letters to lowercase,
 *        and then removes all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *        Finally, it inserts a "." before each consonant and prints the resulting string.
 *
 * Algorithm: String processing with character-by-character transformation and filtering.
 * Time Complexity: O(L) where L is the length of the input string, due to single pass
 *                  iteration and constant time character operations.
 * Space Complexity: O(L) to store the input string and potentially the output string (in C-style char array).
 */

#include <cctype>
#include <cstdio>

using namespace std;

int main() {
  /**
   * Functional Utility: Coordinates the string processing logic, including
   *                     input, character transformation, filtering, and output.
   */
  char x[101];    // Declare a character array to store the input string. Max length 100 + null terminator.
  scanf("%s", &x); // Read the input string from standard input.

  /**
   * Block Logic: Iterates through each character of the input string to apply transformation and filtering.
   * Pre-condition: `x` contains the input string.
   * Invariant: Each character `x[i]` is processed; loop continues until the null terminator is reached.
   */
  for (int i = 0; x[i] != 0; i++) {
    x[i] = tolower(x[i]); // Convert the current character to lowercase.
    // Check if the current character is a consonant (not a vowel).
    if (x[i] != 'a' && x[i] != 'e' && x[i] != 'i' && x[i] != 'o' &&
        x[i] != 'u' && x[i] != 'y') {
      printf(".%c", x[i]); // Print a "." followed by the consonant.
    }
  }
  return 0; // Indicate successful program termination.
}