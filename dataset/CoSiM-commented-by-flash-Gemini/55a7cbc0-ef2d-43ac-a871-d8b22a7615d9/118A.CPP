/**
 * @file 118A.CPP
 * @brief Solution to Codeforces Problem 118A: "String Task".
 *        The problem involves transforming a given input string according to specific rules:
 *        1. Delete all vowels (A, O, Y, E, U, I), case-insensitive.
 *        2. Insert a "." before each consonant.
 *        3. Convert all consonants to lowercase.
 *
 * Algorithm: String iteration and conditional character processing. The program iterates
 *            through each character of the input string. It explicitly checks for both
 *            uppercase and lowercase vowels to skip them. For consonants, it converts
 *            them to lowercase if necessary and then prints a '.' followed by the consonant.
 * Time Complexity: O(L) where L is the length of the input string, as each character is
 *                  processed a constant number of times.
 * Space Complexity: O(1) beyond input storage, as output is streamed directly.
 */
#include <iostream>
using namespace std;

int main() {
  /**
   * Functional Utility: Declaration of variables for string processing.
   * `s`: Stores the input string.
   * `len`: Stores the length of the input string.
   */
  string s;
  cin >> s; // Input: Read the input string.
  int len = s.length(); // Get the length of the input string.

  /**
   * Block Logic: Iterates through each character of the input string `s`.
   * Pre-condition: `s` contains the input string.
   * Invariant: `i` is the current character index. Characters are processed one by one.
   */
  for (int i = 0; i < len; i++) {
    /**
     * Conditional Logic: Checks if the current character is an uppercase or lowercase vowel.
     * The `int i = 0;` inside this block is redundant and has no functional effect outside its scope.
     * If it is a vowel, it is skipped (`continue`).
     */
    if (s[i] == 'a' || s[i] == 'y' || s[i] == 'Y' || s[i] == 'e' ||
        s[i] == 'i' || s[i] == 'o' || s[i] == 'u' || s[i] == 'A' ||
        s[i] == 'E' || s[i] == 'I' || s[i] == 'O' || s[i] == 'U') {
      int i = 0; // Redundant statement, does not affect the outer loop's 'i'.
    }
    /**
     * Block Logic: If the character is not a vowel (i.e., it's a consonant).
     * Pre-condition: `s[i]` is a consonant.
     * Post-condition: A '.' is printed, followed by the lowercase version of `s[i]`.
     */
    else {
      /**
       * Conditional Logic: Checks if the consonant is an uppercase letter.
       * If it is, convert it to lowercase by adding 32 (ASCII difference between 'A' and 'a').
       */
      if (s[i] >= 65 && s[i] <= 90) { // ASCII values for 'A' to 'Z'.
        s[i] = s[i] + 32; // Convert to lowercase.
        cout << "." << s[i]; // Print '.' and the lowercase consonant.
      }
      /**
       * Conditional Logic: If the consonant is already lowercase.
       */
      else
        cout << "." << s[i]; // Print '.' and the lowercase consonant as is.
    }
  }

  return 0;
}
