/**
 * @file 118A.CPP
 * @brief Solution to Codeforces Problem 118A: "String Task".
 *        The problem involves transforming a given input string according to specific rules:
 *        1. Delete all vowels (A, O, Y, E, U, I), case-insensitive.
 *        2. Insert a "." before each consonant.
 *        3. Convert all consonants to lowercase.
 *
 * Algorithm: String iteration and conditional character processing. The program iterates
 *            through each character of the input string. It checks if the character is a vowel.
 *            If it is, the character is skipped. If it's a consonant, a '.' is appended
 *            to the result string, followed by the lowercase version of the consonant.
 * Time Complexity: O(L) where L is the length of the input string, as each character is
 *                  processed a constant number of times.
 * Space Complexity: O(L) to store the transformed output string.
 */
#include <cctype>
#include <iostream>
#include <string>

int main() {
  /**
   * Functional Utility: Declaration of variables for string processing.
   * `slowo`: Stores the input string.
   * `dlugosc`: Stores the length of the input string.
   * `output`: Stores the transformed output string.
   * `c`: Temporary character variable for case conversion.
   */
  std::string slowo;
  std::cin >> slowo; // Input: Read the input string.
  int dlugosc = slowo.length(); // Get the length of the input string.
  std::string output = "";    // Initialize an empty string for the transformed output.
  char c;

  /**
   * Block Logic: Iterates through each character of the input string `slowo`.
   * Pre-condition: `slowo` contains the input string. `output` is empty.
   * Invariant: `i` is the current character index. `output` accumulates transformed characters.
   */
  for (int i = 0; i < dlugosc; i++) {
    /**
     * Conditional Logic: Checks if the current character is an uppercase vowel (A, O, Y, E, U, I)
     * using their ASCII values.
     * If it is a vowel, it is skipped (`continue`).
     */
    if (slowo[i] == 65 || slowo[i] == 79 || slowo[i] == 89 || slowo[i] == 69 ||
        slowo[i] == 85 || slowo[i] == 73) {
      continue;
    }
    /**
     * Conditional Logic: Checks if the current character is a lowercase vowel (a, o, y, e, u, i)
     * using their ASCII values.
     * If it is a vowel, it is skipped (`continue`).
     */
    else if (slowo[i] == 97 || slowo[i] == 111 || slowo[i] == 121 ||
               slowo[i] == 101 || slowo[i] == 117 || slowo[i] == 105) {
      continue;
    }
    /**
     * Block Logic: If the character is not a vowel (i.e., it's a consonant).
     * Pre-condition: `slowo[i]` is a consonant.
     * Post-condition: `output` has "." and the lowercase version of `slowo[i]` appended.
     */
    else {
      output += ".";            // Append a dot before the consonant.
      c = slowo[i];             // Get the character.
      output += tolower(c);     // Convert to lowercase and append to output.
    }
  }

  // Output: Print the final transformed string.
  std::cout << output;

  return 0;
}