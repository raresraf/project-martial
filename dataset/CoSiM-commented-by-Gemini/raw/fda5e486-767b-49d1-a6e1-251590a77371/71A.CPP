/**
 * @file 71A.cpp
 * @brief A straightforward solution to the "Way Too Long Words" problem.
 * This program reads a number of words and, for each word, if its length is
 * greater than 10, it prints an abbreviation of it. Otherwise, it prints the
 * word as is. The abbreviation consists of the first character, the number of
 * characters between the first and last characters, and the last character.
 *
 * Algorithm:
 * 1. Read the number of words `n`.
 * 2. Loop `n` times:
 *    a. Read a word.
 *    b. If the word's length is greater than 10, print the abbreviation.
 *    c. Otherwise, print the word as is.
 *
 * Time Complexity: O(N * L), where N is the number of words and L is the length of the longest word.
 * Space Complexity: O(L) for storing the longest word.
 */
#include <iostream>
using namespace std;
string a;
int n;
int main() {
  // Read the number of words to process.
  cin >> n;
  // Loop n times to process each word.
  for (int i = 0; i < n; i++) {
    cin >> a;
    // If the word's length is greater than 10, abbreviate it.
    if (a.length() > 10) {
      cout << a[0] << a.length() - 2 << a[a.length() - 1] << endl;
    } else {
      // Otherwise, print the word as is.
      cout << a << endl;
    }
  }
}