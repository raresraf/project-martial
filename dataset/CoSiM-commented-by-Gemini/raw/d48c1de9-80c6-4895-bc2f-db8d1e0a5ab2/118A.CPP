/*
~ Author    : @tridib_2003
~ Title     : 118A. String Task
~ Link      : https://codeforces.com/problemset/problem/118/A
*/

/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem (Codeforces 118A).
 * @details This program processes a string by deleting vowels, inserting a '.'
 * before consonants, and converting consonants to lowercase. This version
 * processes the string character-by-character and prints the output directly
 * without building an intermediate result string.
 *
 * @note The case-insensitive vowel check uses an obscure trick (`'a' - ' '`)
 * to represent uppercase vowels via ASCII arithmetic, which is not standard
 * or portable practice.
 *
 * @section algorithm
 * Algorithm: Direct Character Processing
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. Use a switch statement to check if the character is a vowel (both upper
 *    and lower case).
 * 4. If the character is a vowel, ignore it and continue to the next character.
 * 5. If it is a consonant, print a '.' followed by the lowercase version of
 *    the consonant.
 *
 * @section complexity
 * - Time Complexity: O(L), for the single pass through the string of length L.
 * - Space Complexity: O(L), to store the input string. Auxiliary space is O(1).
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  /**
   * @brief Optimizes C++ standard streams for faster I/O in competitive programming.
   */
  ios::sync_with_stdio(false);
  cin.tie(0);
  /*
      int tc;
  cin>>tc;
  while(tc--) {
      }
      */

  string str;
  cin >> str;

  /**
   * @brief Iterates through the input string, processing one character at a time.
   */
  for (int i = 0; i < str.length(); i++) {
    /**
     * @brief Checks if the character is a vowel (case-insensitive) and skips it.
     * Otherwise, processes it as a consonant.
     */
    switch (str[i]) {
    // Lowercase vowels
    case 'a':
    case 'e':
    case 'i':
    case 'o':
    case 'u':
    case 'y':
    // Uppercase vowels checked via ASCII arithmetic trick.
    // e.g., 'a' (97) - ' ' (32) = 65, which is the ASCII for 'A'.
    case 'a' - ' ':
    case 'e' - ' ':
    case 'i' - ' ':
    case 'o' - ' ':
    case 'u' - ' ':
    case 'y' - ' ':
      continue; // Skips vowels.
    default:
      // For consonants, prints a '.' and the lowercase version of the char.
      cout << "." << (char)tolower(str[i]);
    }
  }
  return 0;
}
