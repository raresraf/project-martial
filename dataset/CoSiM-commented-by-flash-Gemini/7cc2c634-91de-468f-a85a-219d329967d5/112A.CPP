/**
 * @file 112A.CPP
 * @brief Lexicographical string comparison after case normalization.
 * Algorithm: Compares two input strings after converting all uppercase letters to lowercase.
 * Time Complexity: O(N), where N is the length of the shorter string (due to character-by-character processing).
 * Space Complexity: O(N) for storing the input strings.
 */
#include <algorithm>
#include <iostream>
using namespace std;
string a, b;
int main() {
  // Block Logic: Reads two strings from standard input for comparison.
  cin >> a >> b;
  // Block Logic: Normalizes both strings to lowercase for case-insensitive comparison.
  // Invariant: After this loop, both strings 'a' and 'b' will contain only lowercase characters.
  for (int i = 0; i < a.size(); i++) {
    if (a[i] < 92) {
      a[i] += 32;
    }
    if (b[i] < 92) {
      b[i] += 32;
    }
  }
  // Block Logic: Compares the normalized strings lexicographically and prints the result.
  // Functional Utility: Outputs -1 if string 'a' is lexicographically smaller, 1 if larger, and 0 if equal.
  if (a < b) {
    cout << -1;
  } else if (a > b) {
    cout << 1;
  } else if (a == b) {
    cout << 0;
  }
}