/**
 * @file 118A.CPP
 * @brief This program transforms an input string by converting all characters to lowercase,
 * removing vowels (both uppercase and lowercase), and prefixing each remaining consonant with a dot.
 * Algorithm: Iterative string processing with character-by-character transformation and explicit vowel checking.
 * Time Complexity: O(N * M), where N is the length of the input string and M is the number of vowels to check.
 * Space Complexity: O(M) for storing the vowel array, plus O(N) for the input string.
 */

#include <bits/stdc++.h> // Includes most standard C++ library headers.

using namespace std;

#define ll long long // Defines a macro 'll' as a shorthand for 'long long'.

// Global array containing both lowercase and uppercase vowels for efficient checking.
char v[12] = {'a', 'o', 'y', 'e', 'u', 'i', 'A', 'O', 'Y', 'E', 'U', 'I'};

/**
 * @brief Converts an input character to its lowercase equivalent if it's an uppercase letter.
 * @param t The character to be potentially converted.
 * @return The lowercase version of the character, or the original character if it was not uppercase.
 */
char down(char t) {
  if (t <= 'Z') // Checks if the character is an uppercase letter (ASCII values).
    return t + 32; // Converts to lowercase by adding 32 (ASCII difference).
  else
    return t; // Returns the character as is if it's already lowercase or not a letter.
}

int main() {
  string s; // Stores the input string.
  // Reads the input string from standard input.
  cin >> s;
  bool flag; // Flag to indicate if the current character is a consonant (true) or a vowel (false).
  /**
   * Block Logic: Iterates through each character of the input string 's' to apply transformations.
   * For each character, it first determines if it's a vowel or a consonant.
   * If it's a consonant, it's converted to lowercase and printed with a preceding dot.
   */
  for (int i = 0; i < s.length(); i++) {
    flag = true; // Assumes the character is a consonant until proven otherwise.
    /**
     * Block Logic: Checks if the current character 's[i]' is present in the global vowel array 'v'.
     * If a match is found, the character is a vowel, and the 'flag' is set to false.
     */
    for (int j = 0; j < 12; j++) {
      if (s[i] == v[j]) {
        flag = false; // Current character is a vowel.
        break;        // No need to check further, move to the next character in 's'.
      }
    }
    /**
     * Conditional Logic: If 'flag' is true, the character is a consonant.
     * It then prints a dot followed by the lowercase version of the consonant.
     */
    if (flag) {
      cout << "." << down(s[i]);
    }
  }
  // Standard program exit.
  return 0;
}
