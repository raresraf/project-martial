/**
 * @file 266A.CPP
 * @brief Counts the number of adjacent identical characters in a given string.
 *        The 'test' variable appears to specify the effective length of the string
 *        to consider for comparisons, potentially redundant with string.size().
 * Algorithm: Linear scan. Iterates through the string and compares each character
 *            with its immediate successor.
 * Time Complexity: O(L) where L is the length of the string, as it involves a single pass.
 * Space Complexity: O(L) for storing the input string, or O(1) if input string storage
 *                   is not counted (e.g., in-place processing).
 */
#include <iostream>
#include <string> // Required for string manipulation

using namespace std;

/**
 * @brief Main function to count adjacent duplicate characters.
 * Functional Utility: Reads a string and an integer, then computes and
 *                     prints the total count of adjacent identical characters.
 */
int main() {
  string word; // The input string to be analyzed.
  int test;    // Integer, likely representing the length or a limit related to string length.
               // Used here to define the upper bound of the loop (test - 1).

  // Block Logic: Reads the integer 'test' and the string 'word' from standard input.
  cin >> test >> word;

  auto count = 0; // Variable to store the number of adjacent duplicate characters found.

  // Block Logic: Iterates through the string, comparing each character with its next character.
  // Pre-condition: 'word' is a valid string and 'test' is less than or equal to word.size().
  // Invariant: 'count' accumulates the number of times `word[i]` is identical to `word[i + 1]`.
  for (auto i = 0; i < test - 1; i++) {
    if (word[i] == word[i + 1]) {
      count++; // Increment count if adjacent characters are the same.
    }
  }

  // Functional Utility: Prints the final count of adjacent duplicate characters.
  cout << count << endl;

  return 0; // Indicates successful program termination.
}
