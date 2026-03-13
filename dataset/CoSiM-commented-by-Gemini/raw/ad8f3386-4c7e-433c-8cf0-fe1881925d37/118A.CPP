/**
 * @file 118A.CPP
 * @brief Processes a string by removing vowels and modifying consonants.
 *
 * This program reads a string, removes all vowels (a, e, i, o, u, y),
 * inserts a '.' character before each consonant, and converts all consonants
 * to lowercase. The resulting string is constructed but not printed to the
 * standard output.
 *
 * @see https://codeforces.com/problemset/problem/118/A
 */
#include <iostream>
#include <string>

using namespace std;

int main() {
  string s, s2;
  cin >> s;
  int l = s.length();

  /**
   * @brief Main processing loop.
   *
   * This loop iterates through each character of the input string `s`.
   * It checks if the character is a vowel (case-insensitive). If it is, the
   * character is skipped. Otherwise, a '.' is appended to the result string
   * `s2`, followed by the lowercase version of the consonant.
   */
  for (int i = 0; i < l; i++) {
    if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' ||
        s[i] == 'u' || s[i] == 'y' || s[i] == 'A' || s[i] == 'E' ||
        s[i] == 'I' || s[i] == 'O' || s[i] == 'U' || s[i] == 'Y') {
      continue;
    }
    s2 += ".";
    s2 += tolower(s[i]);
  }
  // Note: The resulting string s2 is not printed to the console.
  return 0;
}