/**
 * @file 118A.CPP
 * @brief Solution for Codeforces Problem 118A, "String Task".
 * @details This program processes a string by removing vowels, converting all
 * letters to lowercase, and inserting a '.' before each consonant.
 *
 * Algorithm:
 * The code implements a two-pass approach after an initial transformation.
 * 1. The entire input string is converted to lowercase using `std::transform`.
 * 2. In the first pass, the code iterates through the lowercase string. Using a
 *    helper function `vowel`, it identifies all consonants and stores them in
 *    a character vector.
 * 3. In the second pass, it iterates through the vector of consonants and prints
 *    each one, prepended with a '.' character.
 *
 * Time Complexity: O(N), where N is the length of the string, due to the
 * transformations and loops.
 * Space Complexity: O(N) to store the intermediate vector of consonants.
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long int

/**
 * @brief Checks if a character is a lowercase vowel.
 * @param c The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool vowel(char c) {
  switch (c) {
  case 'a':
  case 'e':
  case 'i':
  case 'o':
  case 'u':
  case 'y':
    return true;
  default:
    return false;
  }
}

int main() {
  string s;
  cin >> s;
  
  // Convert the entire string to lowercase for case-insensitive processing.
  transform(s.begin(), s.end(), s.begin(), ::tolower);
  
  vector<char> m;
  /**
   * @brief First pass: Identify all consonants and store them in a vector.
   */
  for (int i = 0; i < s.length(); i++) {
    if (!vowel(s[i]))
      m.push_back(s[i]);
  }

  /**
   * @brief Second pass: Print the final formatted string from the consonant vector.
   */
  for (int i = 0; i < m.size(); i++) {
    cout << "." << m[i];
  }
}
