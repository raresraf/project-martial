/**
 * @file 118A.CPP
 * @brief A correct solution for the "String Task" problem (Codeforces 118A).
 *
 * @algorithm
 * The problem requires processing a string with the following rules:
 * 1. Delete all vowels ('A', 'O', 'Y', 'E', 'U', 'I').
 * 2. Before each consonant, insert a ".".
 * 3. All uppercase consonants should be replaced with their corresponding lowercase ones.
 *
 * This solution implements this by iterating through the input string character
 * by character and building a new result string.
 * 1. Read the input string `s`.
 * 2. Create an empty result string `n`.
 * 3. For each character in `s`:
 *    a. Check if it is a vowel (either uppercase or lowercase). If so, ignore it
 *       and continue to the next character.
 *    b. If it is a consonant, first append a '.' to `n`.
 *    c. Convert the consonant to lowercase if it's uppercase.
 *    d. Append the lowercase consonant to `n`.
 * 4. Print the final result string `n`.
 *
 * @complexity
 * - Time Complexity: O(L), where L is the length of the input string, because
 *   it performs a single pass over the string.
 * - Space Complexity: O(L), as a new string `n` is created which can have a
 *   length up to 2*L in the worst case (all consonants).
 */
#include <bits/stdc++.h>
#include <iostream>
#include <string>
using namespace std;
int main() {
  string s, n; // `s` for input, `n` for the result string.
  n = "";
  cin >> s;
  //    cout<<s<<endl;
  /**
   * @brief Main loop iterates through each character of the input string `s`.
   */
  for (int i = 0; i < s.length(); i++) {
    /**
     * @brief Checks if the current character is a vowel (case-insensitive).
     * If it is a vowel, the character is ignored, and the loop continues.
     */
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue;
    }

    // Block for processing consonants.
    // Prepend a '.' to the result string.
    n = n + '.';
    
    // Check if the consonant is uppercase.
    if (s[i] >= 'A' && s[i] <= 'Z')
      // Convert to lowercase by adding 32 (ASCII difference).
      // Note: A more portable way is to use `tolower()`.
      s[i] = s[i] + 32;
    // Append the lowercase consonant to the result string.
    n = n + s[i];
  }

  cout << n << endl;
  return 0;
}
