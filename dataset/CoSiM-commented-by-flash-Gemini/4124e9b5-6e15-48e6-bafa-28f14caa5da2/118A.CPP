/**
 * @file 118A.CPP
 * @brief Transforms an input string by printing only its lowercase consonants, each prefixed with a dot.
 *
 * This program reads a single string from standard input. It then iterates
 * through each character of the string, converts it to its lowercase equivalent,
 * and checks if it is a vowel (a, e, i, o, u, y). If the character is a vowel,
 * it is skipped. Otherwise, a dot '.' followed by the lowercase consonant
 * is printed to standard output.
 *
 * Algorithm: Character-by-character string traversal, case conversion, and conditional filtering.
 * Time Complexity: O(L) where L is the length of the input string. The program
 *                  iterates through each character of the string once, performing
 *                  constant time operations (case conversion, comparison, and printing).
 * Space Complexity: O(1) beyond the storage required for the input string itself,
 *                   as only a few scalar variables are used during processing.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Main function to process the input string and print transformed output.
   *
   * Reads a string, then iterates through its characters. For each character,
   * it converts to lowercase and checks if it's a vowel. Consonants are printed
   * with a preceding dot, while vowels are skipped.
   */

  // @param input Stores the string read from standard input.
  string input;
  // Reads a single word (token) from standard input into `input`.
  cin >> input;

  /**
   * @brief Loop to iterate through each character of the input string.
   *
   * For each character, it converts to lowercase and checks if it is a vowel.
   * Consonants are printed prefixed with a dot, vowels are ignored.
   * Invariant: All characters up to index `i-1` have been processed according to the rules.
   */
  for (int i = 0; i < input.size(); ++i) {
    // @param elementAti Stores the current character from the input string, converted to lowercase.
    char elementAti = tolower(input[i]); // Convert current character to lowercase.

    // @brief Conditional check to filter out vowels and process consonants.
    // Logic: If the lowercase character is one of 'a', 'e', 'i', 'o', 'u', 'y', it's a vowel and skipped.
    //        Otherwise, it's a consonant, and a dot followed by the character is printed.
    if (elementAti == 'a' || elementAti == 'e' || elementAti == 'i' ||
        elementAti == 'o' || elementAti == 'u' || elementAti == 'y')
      continue; // Skip vowels.
    else
      cout << "." << elementAti; // Print a dot followed by the consonant.
  }

  // @brief Prints a newline character at the end of the output.
  cout << endl;
}