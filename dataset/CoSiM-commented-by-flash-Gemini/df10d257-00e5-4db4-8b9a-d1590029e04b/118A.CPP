/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 *
 * This program transforms an input string according to specific rules:
 * 1. All vowels (a, o, y, e, u, i, case-insensitive) are removed.
 * 2. A '.' character is inserted before each consonant.
 * 3. All resulting characters are converted to lowercase.
 *
 * Algorithm: Character-by-character string processing.
 * Time Complexity: O(L) where L is the length of the input string, due to single pass iteration.
 * Space Complexity: O(L) for the output string, potentially O(1) if printed directly.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>

using namespace std;

typedef long long ll;

/**
 * @brief Checks if a given character is a vowel (case-insensitive).
 *
 * @param c The character to check.
 * @return 1 if the character is a vowel (a, o, y, e, u, i), 0 otherwise.
 * Functional Utility: Provides a utility to classify characters for string transformation.
 */
int isvowel(int c) {
  c = tolower(c); // Convert character to lowercase for case-insensitive comparison.
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

/**
 * @brief Main function to perform the string transformation.
 *
 * Functional Utility: Reads the input string, iterates through its characters,
 * and prints the transformed string to standard output.
 */
int main() {
  string s;
  cin >> s; // Read the input string.
  /**
   * Block Logic: Iterates through each character of the input string.
   * For each character, it checks if it's a consonant using the `isvowel` helper function.
   * If it's a consonant, a '.' is printed, followed by the lowercase version of the consonant.
   * Invariant: All processed characters are either skipped (vowels) or transformed (consonants).
   */
  for (auto c : s) {
    if (!isvowel(c)) { // Check if the character is a consonant.
      printf(".%c", tolower(c)); // Print '.' and the lowercase consonant.
    }
  }
  cout << '\n'; // Print a newline character at the end.
}
