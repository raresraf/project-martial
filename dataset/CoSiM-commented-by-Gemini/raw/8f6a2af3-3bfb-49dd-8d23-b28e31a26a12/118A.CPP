/**
 * @file 118A.CPP
 * @brief Solution to the "String Task" problem.
 * @see https://codeforces.com/problemset/problem/118/A
 *
 * This program processes a string according to a specific set of rules:
 * 1. It deletes all the vowels.
 * 2. It inserts a character "." before each consonant.
 * 3. It replaces all uppercase consonants with their corresponding lowercase ones.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Iterate through each character of the string.
 * 3. If the character is a vowel, ignore it.
 * 4. If the character is a consonant, print a "." followed by the lowercase
 *    version of the consonant.
 *
 * Time Complexity: O(N), where N is the length of the string.
 * Space Complexity: O(N) to store the input string.
 */

#include <cctype>
#include <cstring>
#include <iostream>

using namespace std;

/**
 * @brief Checks if a character is a vowel.
 * @param c The character to check.
 * @return True if the character is a vowel, false otherwise.
 */
bool isVowel(char c) {
  return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' ||
          c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' ||
          c == 'y' || c == 'Y');
}

int main() {

  string word;
  cin >> word;

  // Invariant: Iterates through each character of the input string to apply
  // the transformation rules.
  for (char c : word) {

    // Pre-condition: If the character is a vowel, it is skipped.
    if (isVowel(c)) {
      continue;
    }

    cout << '.';

    // Block Logic: If the character is an uppercase consonant, it is converted
    // to lowercase before being printed.
    if (isupper(c)) {
      char x = c + 32;
      cout << x;
    } else
      cout << c;
  }

  cout << endl;

  return 0;
}