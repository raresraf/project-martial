/**
 * @file 118A.CPP
 * @brief Processes an input word by converting to lowercase, removing vowels, and formatting consonants with preceding dots.
 *
 * Problem Name: String Task (Codeforces 118A)
 * Algorithm: Iterates through the input string, converts characters to lowercase, filters out vowels,
 *            and stores consonants (prefixed with a dot) in a vector for final output.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(L) for storing the processed characters in a vector.
 */
#include "bits/stdc++.h" // Includes most standard C++ libraries, e.g., <iostream>, <string>, <vector>, <algorithm>.
// PRAY :DOLPHINGARLIC: orz orz orz orz orz
// PRAY :DORI: orz orz orz orz orz orz orz
// PRAY :SAHIL KUCHLOUS: orz orz orz orz orz
// pray :stefan: orz orz orz orz orz orz orz
// pray :foshy: orz orz orz orz orz orz orz
#define GS                                                                     \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(nullptr);                                                            \
  cout.tie(0) // Functional Utility: Optimizes I/O operations by decoupling C++ streams
              // from C stdio and untying cin from cout. Also sets cout precision for speed.
#define all(x) x.begin(), x.end() // Functional Utility: Shorthand for passing begin and end iterators to algorithms.
#define el '\n'                   // Functional Utility: Shorthand for newline character, often for faster I/O.
#define Int long long             // Functional Utility: Defines 'Int' as a shorthand for 'long long'.
#define elif else if              // Functional Utility: Shorthand for 'else if'.
#define test                                                                   \
  Int t;                                                                       \
  cin >> t;                                                                    \
  while (t--) // Functional Utility: Macro for handling multiple test cases in competitive programming.
// #define - This line appears to be an incomplete macro or leftover, it will be kept due to no code mutation rule.

using namespace std;

/**
 * @brief Main function to process the input string as per String Task rules.
 * @pre Input consists of a single string.
 * @post Prints the processed string with lowercase consonants, each preceded by a dot.
 */
int main() {
  GS; // Functional Utility: Calls the I/O optimization macro.

  string s; // Variable to store the input string.
  // Functional Utility: Reads the input string from standard input.
  cin >> s;

  vector<char> v; // Vector to store the processed characters (dots and consonants).

  /**
   * Block Logic: Iterates through each character of the input string to apply transformations.
   * Invariant: For each character, it's first converted to lowercase. If it's a consonant,
   *            a dot and then the consonant itself are added to the 'v' vector.
   */
  for (int i{}; i < (int)s.size(); i++) {
    s[i] = tolower(s[i]); // Converts the current character to lowercase.

    /**
     * Block Logic: Checks if the current (lowercase) character is a consonant.
     * Pre-condition: `s[i]` is a lowercase character.
     * Invariant: If `s[i]` is not one of the specified vowels ('a', 'o', 'y', 'e', 'u', 'i'), it's treated as a consonant.
     */
    if (s[i] != 'a' && s[i] != 'o' && s[i] != 'y' && s[i] != 'e' &&
        s[i] != 'u' && s[i] != 'i') {
      v.push_back('.');    // Adds a dot.
      v.push_back(s[i]); // Adds the consonant.
    }
  }

  /**
   * Block Logic: Prints all characters stored in the vector 'v'.
   * Invariant: Each character from the vector is printed sequentially.
   */
  for (auto i : v) { // Iterates through the vector 'v'.
    cout << i;     // Prints the current character.
  }
  cout << endl; // Prints a newline character at the end of the output.

  // Indicates successful program execution.
  return 0;
}
/*
12:43 AM
9/18/2020
*/
