/**
 * @file 118A.CPP
 * @brief Solves the "String Task" problem.
 * @details This program processes a given string by performing three main operations:
 * 1. It converts all uppercase letters to their lowercase equivalents.
 * 2. It removes all vowels (a, e, i, o, u, y).
 * 3. It inserts a '.' character before each of the remaining characters (consonants).
 *
 * Algorithm: String manipulation and filtering.
 * Time Complexity: O(N) where N is the length of the input string, as it involves a few linear passes over the string.
 * Space Complexity: O(N) to store the modified string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  string s, a = "";
  cin >> s;
  /**
   * @brief First pass: Normalize and filter the string.
   * This loop iterates through the input string `s`. It converts each character
   * to lowercase and then checks if it's a vowel. Consonants are appended
   * to a new string `a`.
   */
  for (int i = 0; i < s.size(); i++) {
    // Converts character to lowercase
    if (s[i] >= 'A' && s[i] <= 'Z')
      s[i] = s[i] - 'A' + 'a';
    // Filters out vowels
    if (s[i] != 'a' && s[i] != 'e' && s[i] != 'i' && s[i] != 'o' &&
        s[i] != 'u' && s[i] != 'y')
      a += s[i];
  }
  /**
   * @brief Second pass: Constructs the final output string.
   * This loop iterates through the consonant-only string `a` and prints
   * each character prefixed with a '.'.
   */
  for (int i = 0; i < a.size(); i++)
    cout << "." << a[i];
  printf("\n");
  return 0;
}
