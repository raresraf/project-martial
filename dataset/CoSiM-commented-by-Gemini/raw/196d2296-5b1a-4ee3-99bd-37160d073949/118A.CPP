/**
 * @file 118A.CPP
 * @brief Solution for a string processing task, often called "String Task".
 * @details This program processes a given string according to specific rules:
 *          it removes all vowels, inserts a "." character before each consonant,
 *          and converts all uppercase consonants to their lowercase equivalents.
 *
 * Algorithm:
 * 1. Read an input string 'a'.
 * 2. Iterate through each character of the string.
 * 3. For each character, check if it is a vowel (A, E, I, O, U, Y, case-insensitive).
 * 4. If the character is not a vowel (i.e., it's a consonant or another symbol):
 *    a. Convert the character to lowercase if it's uppercase.
 *    b. Print a "." followed by the lowercase character.
 *
 * Time Complexity: O(L), where L is the length of the input string.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  // v, c, con, t are unused variables.
  int v = 0, c = 0, l, con = 0, t;
  // 'a' stores the input string, 'b' is unused.
  string a, b;
  cin >> a;
  // 'l' stores the length of the input string.
  l = a.length();

  /**
   * Block Logic: Iterates through the input string to process each character.
   * Invariant: The output stream contains the processed version of the prefix
   * of the string that has been scanned so far.
   */
  for (int i = 0; i < l; i++) {
    /**
     * Pre-condition: Check if the character is a consonant. The vowels are
     * defined as 'A', 'E', 'I', 'O', 'U', 'Y' in both upper and lower case.
     */
    if (a[i] != 'A' && a[i] != 'E' && a[i] != 'I' && a[i] != 'O' &&
        a[i] != 'U' && a[i] != 'Y' && a[i] != 'a' && a[i] != 'e' &&
        a[i] != 'i' && a[i] != 'o' && a[i] != 'u' && a[i] != 'y') {
      /**
       * Block Logic: If the character is an uppercase consonant, convert it to
       * lowercase. The ASCII value of uppercase letters is less than 97.
       */
      if (a[i] < 97) {
        a[i] = a[i] + 32;
        cout << "." << a[i];
      } else {
        cout << "." << a[i];
      }
    }
    // The variable 'c' is incremented but its value is never used.
    c++;
  }
  return 0;
}
