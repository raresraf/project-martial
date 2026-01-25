/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and prefixing consonants with a dot.
 *
 * This program takes a string as input, converts it to lowercase, and then transforms it
 * according to a specific rule: all vowels ('a', 'e', 'i', 'o', 'u', 'y') are deleted,
 * and each consonant is preceded by a ".".
 *
 * @b Algorithm: The program reads a string, converts it to lowercase, and then iterates
 * through each character. If a character is a vowel, it is ignored. If it is a consonant,
 * a '.' character followed by the consonant itself is appended to a new result string.
 *
 * @b Time_Complexity: O(L), where L is the length of the input string, due to the single
 * pass for transformation and another for processing.
 * @b Space_Complexity: O(L), to store both the original and the transformed strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
#ifndef ONLINE_JUDGE
  // This block is for local testing, redirecting standard I/O to files.
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif
  string s, s1;
  cin >> s;
  int l = s.length();
  // Convert the entire input string to lowercase for case-insensitive processing.
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  /**
   * @brief This loop iterates through the lowercase string to build the result string.
   *
   * Pre-condition: `s` is the lowercase version of the input string.
   * Invariant: `s1` contains the processed version of the substring of `s` from index 0 to `i-1`.
   */
  for (int i = 0; i < l; i++) {
    // Block Logic: Check if the current character is a vowel.
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y')
      continue;
    else {
      // If it's a consonant, prepend a '.' and append it to the result string.
      s1.push_back('.');
      s1.push_back(s[i]);
    }
  }
  // Output the final transformed string.
  cout << s1 << "\n";
  return 0;
}
