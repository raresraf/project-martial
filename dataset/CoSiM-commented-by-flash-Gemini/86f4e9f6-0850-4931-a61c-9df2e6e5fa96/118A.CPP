/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem from Codeforces (Problem 118A).
 * This program transforms an input string by converting uppercase letters to lowercase,
 * deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i'), and inserting a "." before each consonant.
 *
 * Algorithm: Character-by-character processing with functional programming style.
 * Each character is converted to lowercase, and then a custom `isConsonant` function
 * is used to determine if a character should be prefixed with a dot and printed.
 * Time Complexity: O(L), where L is the length of the input string, as each character is processed once.
 * Space Complexity: O(L) for storing the input string.
 */

#include <algorithm> // @brief Required for std::for_each.
#include <iostream>  // @brief Required for standard input/output streams (cin, cout).
#include <string>    // @brief Required for std::string.
#include <cctype>    // @brief Required for std::tolower.

using namespace std;

// @brief Global array containing characters considered as vowels for the problem.
// Note: Named 'consonants' but contains vowels for the purpose of the 'isConsonant' function logic.
char consonants[6] = {'a', 'e', 'i', 'o', 'u', 'y'};

bool isConsonant(char a) {
  /**
   * @brief Checks if a given character is a consonant based on the defined 'consonants' array.
   *
   * @param a The character to check.
   * @return True if the character is NOT found in the global 'consonants' array (i.e., it's a consonant
   *         according to this logic), False otherwise (i.e., it's a vowel).
   * @note The global array `consonants` in this context actually contains vowels,
   *       thus this function effectively returns true for non-vowels.
   */
  for (char c : consonants) { // @brief Iterate through the predefined list of vowels.
    if (a == c)
      return false; // Inline: If the character is a vowel, it's not a consonant.
  }
  return true; // Inline: If not found in the vowel list, it's a consonant.
}

int main() {
  /**
   * @brief Main function to perform the string transformation task.
   * Reads an input string, converts all characters to lowercase, and then
   * processes them using `std::for_each` with a lambda to print consonants
   * prefixed with a dot.
   */

  string input; // @brief The input string.

  cin >> input; // @brief Read the input string.

  // Block Logic: Iterate through each character of the input string and apply transformations.
  // Using std::for_each with a lambda to process each character in-place.
  // Pre-condition: 'input' contains the string to be transformed.
  // Invariant: Each character is converted to lowercase, and if it's a consonant
  //            (as determined by isConsonant), it's printed prefixed with '.'.
  for_each(input.begin(), input.end(), [](char &c) {
    c = ::tolower(c); // Inline: Convert the current character to lowercase.
    if (isConsonant(c)) { // Inline: Check if the lowercase character is a consonant.
      cout << '.' << c; // Inline: Print '.' followed by the consonant.
    }
  });

  cout << endl; // @brief Print a newline character at the end of the output.

  return 0; // @brief Indicates successful program termination.
}
