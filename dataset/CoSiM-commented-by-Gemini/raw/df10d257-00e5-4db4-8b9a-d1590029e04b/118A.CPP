#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>

/**
 * @file 118A.CPP
 * @brief Solution for a string processing problem, "String Task".
 *
 * Problem Statement:
 * The program takes a string as input and processes it according to the
 * following rules:
 * 1. All vowels are deleted. The vowels are 'A', 'O', 'Y', 'E', 'U', 'I'.
 * 2. A "." character is inserted before each consonant.
 * 3. All uppercase consonants are replaced with their corresponding lowercase
 * ones.
 *
 * Algorithm:
 * The program iterates through each character of the input string.
 * 1. Read the input string.
 * 2. For each character, check if it is a vowel (case-insensitively,
 *    including 'y').
 * 3. If the character is not a vowel (i.e., it's a consonant), convert it to
 *    lowercase.
 * 4. Print a "." followed by the lowercase consonant.
 * 5. If the character is a vowel, do nothing (effectively deleting it).
 *
 * Time Complexity: O(L), where L is the length of the string.
 * Space Complexity: O(L) to store the input string.
 */

using namespace std;

typedef long long ll;

/**
 * @brief Checks if a character is a vowel (a, e, i, o, u, y),
 * case-insensitively.
 * @param c The character to check.
 * @return Non-zero (true) if the character is a vowel, 0 (false) otherwise.
 */
int isvowel(int c) {
  // Convert character to lowercase for a single set of comparisons.
  c = tolower(c);
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

int main() {
  string s;
  cin >> s;

  // Block Logic: Iterate through each character of the input string.
  for (auto c : s) {
    // Pre-condition: Check if the character is not a vowel.
    if (!isvowel(c)) {
      // If it's a consonant, print a dot followed by its lowercase version.
      // Using printf for formatted output.
      printf(".%c", tolower(c));
    }
  }
  cout << '\n';
}