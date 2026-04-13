/**
 * @file 118A.CPP
 * @brief Solution to a competitive programming problem: "A. String Task".
 *
 * This program transforms an input string by applying a set of rules:
 * 1. All uppercase letters are converted to lowercase.
 * 2. Specific vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
 * 3. A '.' character is inserted before each remaining consonant.
 *
 * Algorithm: Character-by-character processing with case conversion and conditional filtering.
 *            The string is first converted entirely to lowercase, then iterated through
 *            to build the transformed output based on vowel/consonant checks.
 * Time Complexity: O(L), where L is the length of the input string, due to one pass
 *                  for case conversion and another pass for filtering and concatenation.
 * Space Complexity: O(L) for storing the input and transformed strings.
 */
#include <algorithm> // Required for std::transform.
#include <iostream>  // Required for input/output operations (cin, cout, getline).
#include <string>    // Required for std::string.
#include <cctype>    // Required for ::tolower.

/**
 * @brief Standard namespace usage.
 * In competitive programming, 'using namespace std;' is common practice to avoid
 * repeatedly prefixing standard library elements with 'std::'.
 */
using namespace std;

/**
 * @brief Main function: entry point of the program.
 * Reads an input string, converts it to lowercase, and then applies
 * vowel removal and consonant prefixing rules to generate and print
 * a transformed string.
 */
int main() {
  string input; // Stores the input string.
  // Reads an entire line from standard input into the 'input' string.
  getline(cin, input);

  /**
   * @brief Converts all characters in the 'input' string to lowercase.
   * Block Logic: Uses `std::transform` to apply the `::tolower` function to
   * every character in the `input` string, modifying it in place. This ensures
   * a case-insensitive processing for subsequent steps.
   */
  transform(input.begin(), input.end(), input.begin(),
            ::tolower); // changing to lower case

  string output = ""; // Initializes an empty string to store the transformed result.

  /**
   * @brief Iterates through the lowercase input string, filtering vowels and formatting consonants.
   * Block Logic: For each character:
   * 1. If the character is one of the specified vowels ('a', 'o', 'y', 'e', 'u', 'i'), it is skipped.
   * 2. Otherwise (if it's a consonant), a period ('.') is appended to the `output` string,
   *    followed by the consonant itself.
   * Invariant: `output` accumulates the transformed string, containing only consonants,
   *            each preceded by a dot.
   */
  for (int i = 0; i < input.length(); i++) {
    // Checks if the current character is one of the specified vowels.
    if (input.at(i) == 'a' || input.at(i) == 'o' || input.at(i) == 'y' ||
        input.at(i) == 'e' || input.at(i) == 'u' || input.at(i) == 'i') {
      continue; // Skips the current character if it's a vowel.
    } else {
      // If the character is a consonant, prefixes it with a dot and appends to 'output'.
      output = output + "." + input.at(i);
    }
  }

  // Prints the final transformed string.
  cout << output;

  // Standard program exit.
  return 0;
}
