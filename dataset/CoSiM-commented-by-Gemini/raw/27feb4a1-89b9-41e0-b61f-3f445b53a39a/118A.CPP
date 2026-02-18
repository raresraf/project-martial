/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem (118A) from Codeforces.
 * @details This program processes a string by applying three rules:
 * 1. Delete all vowels (A, O, Y, E, U, I), case-insensitively.
 * 2. Insert a character "." before each consonant.
 * 3. Convert all uppercase consonants to lowercase.
 *
 * @author Your Name (or leave blank)
 *
 * @algorithm The solution is a two-phase process: first removing vowels, then
 * processing the remaining consonants.
 * 1. Read the input string.
 * 2. Use the "erase-remove" idiom repeatedly to delete all occurrences of
 *    vowels (both upper and lower case, including 'y'/'Y') from the string.
 * 3. Iterate through the modified string, which now only contains consonants.
 * 4. For each character:
 *    a. If it's an uppercase letter, convert it to lowercase.
 *    b. Print a '.' character.
 *    c. Print the lowercase consonant.
 *
 * @complexity
 * Time Complexity: O(N), where N is the length of the string. The multiple erase-remove
 * passes and the final loop are all linear in the length of the string.
 * Space Complexity: O(N) to store the input string.
 */
#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;

  getline(cin, s);

  /**
   * @block
   * @brief Removes all vowels (case-insensitive, including 'y') from the string.
   * @details This uses the erase-remove idiom for each vowel. `remove` shifts all non-vowel
   * elements to the front and returns an iterator to the new logical end. `erase` then
   * truncates the string at that point.
   */
  s.erase(remove(s.begin(), s.end(), 'a'), s.end());
  s.erase(remove(s.begin(), s.end(), 'e'), s.end());
  s.erase(remove(s.begin(), s.end(), 'i'), s.end());
  s.erase(remove(s.begin(), s.end(), 'o'), s.end());
  s.erase(remove(s.begin(), s.end(), 'u'), s.end());
  s.erase(remove(s.begin(), s.end(), 'A'), s.end());
  s.erase(remove(s.begin(), s.end(), 'E'), s.end());
  s.erase(remove(s.begin(), s.end(), 'I'), s.end());
  s.erase(remove(s.begin(), s.end(), 'O'), s.end());
  s.erase(remove(s.begin(), s.end(), 'U'), s.end());
  s.erase(remove(s.begin(), s.end(), 'Y'), s.end());
  s.erase(remove(s.begin(), s.end(), 'y'), s.end());

  /**
   * @block
   * @brief Processes the remaining consonants.
   * @details Iterates through the consonant-only string, converts each character
   * to lowercase, and prints it preceded by a '.'.
   */
  for (int i = 0; i < s.size(); i++) {
    // Check for uppercase using ASCII values and convert to lowercase.
    if (s[i] >= 65 && s[i] <= 90) {
      s[i] = s[i] + 32;
    }

    cout << ".";
    cout << s[i];
  }

  // return 0;
}
