/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 *
 * This program transforms an input string according to specific rules:
 * 1. All vowels (a, o, y, e, u, i, case-insensitive) are removed.
 * 2. A '.' character is inserted before each consonant.
 * 3. All resulting characters are converted to lowercase.
 *
 * Algorithm: Character-by-character string processing and reconstruction.
 * Time Complexity: O(L) where L is the length of the input string, due to single pass iteration.
 * Space Complexity: O(L) for the new string `ss` that stores the transformed output.
 */
#include <iostream>
using namespace std;
/**
 * @brief Main function to perform the string transformation.
 *
 * Functional Utility: Reads the input string, iterates through its characters,
 * filters out vowels, converts consonants to lowercase, prepends '.' to them,
 * and builds a new string which is then printed.
 */
int main() {
  string s, ss = ""; // 's' stores the input string, 'ss' stores the transformed output string.
  cin >> s;          // Read the input string.
  int len = s.length(); // Get the length of the input string.

  /**
   * Block Logic: Iterates through each character of the input string `s`.
   * It identifies vowels (case-insensitive) and skips them.
   * For consonants, it converts them to lowercase if necessary, prepends a '.',
   * and appends them to the `ss` string.
   * Invariant: `ss` accumulates the transformed characters according to the problem rules.
   */
  for (int i = 0; i < len; i++) {
    // Check if the character is a vowel (case-insensitive).
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y')
      continue; // If it's a vowel, skip to the next character.

    ss += '.'; // Prepend a '.' for consonants.
    // Convert consonant to lowercase if it's uppercase, then append to 'ss'.
    if (s[i] >= 'a' && s[i] <= 'z')
      ss += s[i]; // Already lowercase.
    else
      ss += (s[i] - 'A' + 'a'); // Convert uppercase to lowercase.
  }
  cout << ss; // Output the transformed string.
}