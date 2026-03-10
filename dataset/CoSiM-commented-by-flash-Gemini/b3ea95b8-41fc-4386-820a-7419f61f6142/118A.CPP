/**
 * @file 118A.CPP
 * @brief This program implements a string transformation task as part of a competitive programming problem.
 *   The task involves converting an input string by:
 *   1. Converting all uppercase letters to lowercase.
 *   2. Deleting all vowels (a, o, y, e, u, i).
 *   3. Inserting a "." character before each consonant.
 *   Problem description: "String Task", Codeforces 118A.
 * * Algorithm: String processing with character-by-character filtering and transformation.
 *   It iterates through the input string, applies rules to each character (lowercase conversion, vowel/consonant check),
 *   and builds a new sequence of characters according to the rules.
 * Time Complexity: O(L) where L is the length of the input string, as it involves a single pass through the string.
 * Space Complexity: O(L) to store the processed characters in the `vector<char> v`. In the worst case (all consonants), the output string length can be up to `2*L`.
 */
#include <algorithm> // Required for std::tolower (implicitly included by <bits/stdc++.h> if used).
#include <cstring>   // May be included for C-style string functions, but not strictly used here.
#include <ctype.h>   // Required for tolower.
#include <iostream>  // Required for std::cin, std::cout.
#include <string>    // Required for std::string.
#include <vector>    // Required for std::vector.
using namespace std;

typedef long long int lli; // Functional Utility: Alias for long long int.
typedef long long ll;     // Functional Utility: Alias for long long.
typedef unsigned long long int ulli; // Functional Utility: Alias for unsigned long long int.

/**
 * @brief Main function to perform the string transformation.
 * Functional Utility: Configures I/O, reads the input string, applies the transformation rules, and prints the result.
 */
int main(int argc, char const *argv[]) {
  // Functional Utility: Optimizes C++ standard I/O streams for faster execution by
  //                     disabling synchronization with C stdio and untying cin from cout.
  ios_base::sync_with_stdio(0);
  string a, ss;
  // Block Logic: Reads the input string.
  cin >> a;
  // Functional Utility: A vector to store the transformed characters.
  vector<char> v;
  // Inline: `ss = a.size();` is potentially a type mismatch (string = size_t). This line seems to have no functional impact on the program's logic.
  ss = a.size();
  /**
   * Block Logic: Iterates through each character of the input string 'a'.
   * Invariant: In each iteration, a character is processed: converted to lowercase,
   *            checked if it's a vowel, and formatted if it's a consonant.
   */
  for (int i = 0; i < a.size(); ++i) {
    // Block Logic: Converts the current character to lowercase.
    a[i] = tolower(a[i]);
    // The comment `dumb y is not a vowel or what? :(` suggests a common competitive programming
    // specific rule or oversight where 'y' is often considered a vowel for these types of problems.
    /**
     * Block Logic: Checks if the current character is a consonant.
     * Pre-condition: `a[i]` is already in lowercase.
     * Inline: The condition `a[i] != 'a' && ... && a[i] != 'y'` explicitly defines which characters are considered vowels.
     *         If the character is NOT one of these vowels, it is a consonant.
     */
    if (a[i] != 'a' && a[i] != 'e' && a[i] != 'i' && a[i] != 'o' &&
        a[i] != 'u' && a[i] != 'y') {
      // If it's a consonant, prepend a '.'
      v.push_back('.');
      // Then append the consonant itself.
      v.push_back((a[i]));
    }
  }
  /**
   * Block Logic: Iterates through the vector of transformed characters and prints them.
   * Pre-condition: `v` contains the fully processed character sequence.
   */
  for (int i = 0; i < v.size(); ++i) {
    cout << v[i];
  }
  return 0;
}
