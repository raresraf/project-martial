/**
 * @file 71A.CPP
 * @brief Solution to a problem that abbreviates long words.
 *
 * This program reads an integer 'n' representing the number of words to process.
 * For each word, if its length is greater than 10, it is abbreviated to its first character,
 * followed by the count of intermediate characters, and then its last character.
 * Otherwise, the word is printed as is.
 * This is a common string manipulation task often encountered in competitive programming.
 *
 * Algorithm: Iterative processing of strings with conditional abbreviation based on length.
 * Time Complexity: O(N * L), where N is the number of words and L is the maximum string length.
 * Space Complexity: O(L) for storing the input string.
 */
#include <iostream>
using namespace std;
string a; // Global declaration of string 'a' for input, generally avoided in modern C++.
int n;    // Global declaration of int 'n' for the number of words, generally avoided in modern C++.
int main() {
  // Read the total number of words 'n' to be processed.
  cin >> n;
  /**
   * Block Logic: Loop through each of the 'n' words, applying the abbreviation logic.
   * Invariant: In each iteration, one word is read, and either its full form or
   *            abbreviated form is printed.
   */
  for (int i = 0; i < n; i++) {
    // Read the current word into the global string 'a'.
    cin >> a;
    /**
     * Block Logic: Checks the length of the current word to decide on abbreviation.
     * Pre-condition: 'a' contains the word to be processed.
     */
    if (a.length() > 10) {
      // If the word's length is greater than 10, print its abbreviated form:
      // first character, count of middle characters, last character.
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    } else {
      // If the word's length is 10 or less, print it as is.
      cout << a << endl;
    }
  }
  // The 'main' function implicitly returns 0 in C++ if no return statement is present.
}