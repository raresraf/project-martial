/**
 * @file 71A.CPP
 * @brief Solution to the "Way Too Long Words" problem.
 * @details This program implements a string abbreviation algorithm commonly found in competitive
 * programming. If a word's length exceeds 10 characters, it is shortened to a format consisting
 * of its first character, the number of characters between the first and last, and its last character.
 * Otherwise, the word is printed unchanged. This is a typical input processing task designed to
 * test basic string manipulation and conditional logic.
 *
 * @algorithm String Abbreviation
 * @complexity
 * Time Complexity: O(T * L), where T is the number of test cases and L is the maximum length of a word.
 * The complexity is dominated by the I/O operations for reading T lines of strings.
 * Space Complexity: O(L), for storing the longest word string.
 */
#include "bits/stdc++.h"
#define MODULO 1000000007
using namespace std;

typedef unsigned long long ull;
typedef long long ll;

int main() {
  int t;
  cin >> t;
  string s;
  // Block Logic: Process each of the 't' test cases.
  while (t--) {
    cin >> s;
    // Block Invariant: Check if the length of the string is greater than 10.
    // The problem specifies that words "longer than 10" characters should be abbreviated.
    // `s.size() < 11` is equivalent to `s.size() <= 10`.
    if (s.size() < 11)
      cout << s << endl;
    else {
      // Logic: Construct the abbreviation from the first character, the count of
      // intermediate characters (length - 2), and the last character.
      cout << s[0] << s.size() - 2 << s[s.size() - 1] << endl;
    }
  }
  return 0;
}