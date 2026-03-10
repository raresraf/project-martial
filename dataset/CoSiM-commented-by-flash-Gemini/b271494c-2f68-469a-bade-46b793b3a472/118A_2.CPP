/**
 * @file 118A_2.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: Single-pass string processing with filtering and transformation.
 *   It iterates through the input string, converts each character to lowercase (if uppercase),
 *   and then checks if it's a vowel. If it's a consonant, it prints "." followed by the character.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 *   The `strlen(array)` in the loop condition is assumed to be optimized for O(L) overall performance.
 * Space Complexity: O(L) for storing the input character array.
 */
#include <cstring>   // Required for `strlen`.
#include <ctype.h>   // Required for `isupper`, `tolower`.
#include <iomanip>   // Included, but not used.
#include <iostream>  // Required for `std::cin`, `std::cout`.
#include <math.h>    // Included, but not used.
#include <time.h>    // Included, but not used.
using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Reads the input string, processes each character for case conversion,
 *                     vowel filtering, and consonant formatting, then prints the result.
 */
int main() {
  int n; // Declared but unused variable.
  char array[105]; // C-style string to store the input string. Max length 104 characters + null terminator.
  // Block Logic: Reads the input string into the character array `array`.
  cin >> array;
  /**
   * Block Logic: Iterates through each character of the input string `array`.
   * Pre-condition: `array` is populated with the input string.
   * Invariant: Each character is processed for case conversion and vowel/consonant filtering.
   * Inline: `strlen(array)` computes the length of the string.
   */
  for (int i = 0; i < strlen(array); i++) {
    // Conditional Logic: Converts uppercase characters to lowercase in-place.
    if (isupper(array[i])) {
      array[i] = tolower(array[i]);
    }

    /**
     * Conditional Logic: Checks if the current character `array[i]` is a consonant.
     * It explicitly checks against all vowels. If it is NOT a vowel, it is considered a consonant.
     */
    if (array[i] != 'a' && array[i] != 'e' && array[i] != 'o' &&
        array[i] != 'i' && array[i] != 'u' && array[i] != 'y') {
      // If it's a consonant, print a "." followed by the consonant.
      cout << "." << array[i];
    }
  }

  return 0; // Indicates successful program termination.
}
