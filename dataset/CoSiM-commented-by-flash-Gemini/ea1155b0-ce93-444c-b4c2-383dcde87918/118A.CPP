/**
 * @file 118A.CPP
 * @brief This program transforms an input string according to specific rules:
 * 1. All uppercase letters are converted to lowercase.
 * 2. Specific vowels ('a', 'o', 'y', 'e', 'u', 'i') are removed.
 * 3. A '.' character is inserted before each remaining consonant.
 * Algorithm: Case conversion using `std::transform`, followed by iterative character filtering and concatenation.
 * Time Complexity: O(N), where N is the length of the input string, due to two passes.
 * Space Complexity: O(N) for storing the input and transformed strings.
 */
#include <algorithm> // Required for std::transform.
#include <iostream>  // Required for input/output operations (cin, cout, getline).
#include <string>    // Required for std::string.

using namespace std;

int main() {
  string input; // Stores the input string.
  // Reads an entire line from standard input into the 'input' string.
  getline(cin, input);

  // Converts all characters in the 'input' string to lowercase.
  // `::tolower` is used as a function pointer to the global `tolower` function.
  transform(input.begin(), input.end(), input.begin(),
            ::tolower); // changing to lower case

  string output = ""; // Initializes an empty string to store the transformed result.

  /**
   * Block Logic: Iterates through the lowercase 'input' string, filtering characters.
   * Vowels ('a', 'o', 'y', 'e', 'u', 'i') are skipped. Consonants are prefixed with a dot
   * and appended to the 'output' string.
   * Invariant: 'output' contains only consonants from 'input', each preceded by a dot.
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
