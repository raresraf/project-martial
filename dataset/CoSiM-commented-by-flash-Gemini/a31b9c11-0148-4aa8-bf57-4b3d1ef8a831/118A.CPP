/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels ('a', 'o', 'y', 'e', 'u', 'i').
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: Two-pass string processing (one for lowercase conversion, one for filtering) with filtering and transformation.
 *   1. Reads the input string.
 *   2. Converts the entire string to lowercase using `std::transform`.
 *   3. Iterates through the lowercase string. For each character, checks if it's a vowel.
 *      If it's a consonant, prints "." followed by the character.
 * Time Complexity: O(L) where L is the length of the input string, as it involves two linear passes (one for `transform`, one for the loop).
 * Space Complexity: O(L) for storing the input string (modified in-place). The result is printed directly.
 */
#include <bits/stdc++.h> // Includes most standard C++ libraries, such as <iostream>, <string>, <algorithm>, <cctype>.
using namespace std;

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Configures I/O, reads the input string, converts it to lowercase,
 *                     then filters out vowels and prints consonants prefixed with a dot.
 */
int main() {
  // Functional Utility: Optimizes C++ standard I/O streams for faster execution by
  //                     disabling synchronization with C stdio and untying cin from cout.
  ios::sync_with_stdio(false);
  cin.tie(NULL); // Unties cin from cout, potentially speeding up input.
  cout.tie(NULL); // Unties cout from cin, potentially speeding up output.

  string s; // Stores the input string.
  // Block Logic: Reads the input string `s`.
  cin >> s;

  // Functional Utility: Converts all characters in string `s` to lowercase in-place.
  // `transform`: Applies the `tolower` function to each character in the range `s.begin()` to `s.end()`.
  // `s.begin()` as destination: Performs the conversion in-place.
  transform(s.begin(), s.end(), s.begin(), ::tolower);

  /**
   * Block Logic: Iterates through each character of the (now lowercase) string `s` using a range-based for loop.
   * Pre-condition: `s` contains only lowercase characters.
   * Invariant: For each character, if it's a vowel, it's skipped; otherwise, it's printed formatted.
   */
  for (char c : s)
    /**
     * Conditional Logic: Checks if the current character `c` is a vowel.
     * If it is a vowel, the `continue` statement skips to the next iteration, effectively deleting the vowel.
     */
    if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y')
      continue; // Skips vowels.
    else
      // If the character is a consonant, print a "." followed by the consonant.
      cout << "." << c;
  // Block Logic: Prints a newline character at the end of the output.
  cout << "\n";
}