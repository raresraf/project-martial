

/**
 * @file 118A.CPP
 * @brief This program transforms an input string. It iterates through the string,
 * converts non-vowel characters to lowercase (if they are not already) and
 * then prints them prefixed with a dot. Vowels are effectively skipped.
 * Algorithm: Character-by-character processing with conditional case conversion and filtering.
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N) for storing the input string.
 */
#include <cstdlib>  // Included in original, though not explicitly used for this logic.
#include <iostream> // Required for input/output operations (std::cin, std::cout).
#include <string>   // Required for std::string.
#include <cctype>   // Required for std::tolower.

using namespace std;

int main(int argc, char *argv[]) { // main function with command line arguments (unused).
  string t; // Stores the input string.
  // Reads the input string from standard input.
  cin >> t;

  int i = 0; // Loop index for iterating through the string.
  /**
   * Block Logic: Iterates through each character of the input string 't'.
   * For each character, it first checks if it's a vowel (case-insensitive).
   * If it's a consonant, it's converted to lowercase and printed with a preceding dot.
   * Vowels are skipped.
   * Invariant: 'i' tracks the current character being processed.
   */
  while (i < t.length()) {
    // This condition checks if the character is NOT a vowel.
    // The `tolower` function needs to be applied first for a correct case-insensitive check.
    // As written, it checks `t[i]` (original case) against given vowels.
    // If it's an uppercase consonant, it will be treated as a consonant.
    if (!(t[i] == 'a' || t[i] == 'e' || t[i] == 'i' || t[i] == 'o' ||
          t[i] == 'u' || t[i] == 'y')) {
      t[i] = tolower(t[i]); // Converts the character to lowercase.
    }

    // After potential lowercase conversion, this condition again checks if the character is NOT a vowel.
    // If it's a consonant (now guaranteed to be lowercase), it is printed with a preceding dot.
    if (!(t[i] == 'a' || t[i] == 'e' || t[i] == 'i' || t[i] == 'o' ||
          t[i] == 'u' || t[i] == 'y')) {
      cout << "." << t[i];
    }
    i++; // Moves to the next character in the string.
  }
  cout << endl; // Prints a newline character at the end of the output.
  // Standard program exit.
  return 0;
}
