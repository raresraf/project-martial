/**
 * @file 112A.CPP
 * @brief Solution for a competitive programming problem involving lexicographical string comparison, ignoring case.
 * Algorithm: Convert both strings to lowercase and then perform direct lexicographical comparison.
 * Time Complexity: O(N) where N is the length of the strings, due to single pass for case conversion and comparison.
 * Space Complexity: O(1) as modifications are in-place or minimal extra space is used.
 */

#include <bits/stdc++.h>
using namespace std;
int main() {
  // a: First input string for comparison.
  // b: Second input string for comparison.
  string a, b;
  cin >> a >> b;
  // Block Logic: Converts both strings to lowercase for case-insensitive comparison.
  // Invariant: After this loop, both strings 'a' and 'b' will contain only lowercase characters.
  for (int i = 0; i < a.size(); i++) {
    a[i] = tolower(a[i]);
    b[i] = tolower(b[i]);
  }
  // Block Logic: Compares the two strings lexicographically.
  if (a == b)
    cout << "0" << endl; // Output 0 if strings are equal.
  else {
    // Block Logic: Iterates through the strings to find the first differing character
    // and determines lexicographical order.
    for (int i = 0; i < a.size(); i++) {
      if (a[i] < b[i]) {
        cout << "-1" << endl; // Output -1 if string 'a' is lexicographically smaller than 'b'.
        break;
      }
      if (a[i] > b[i]) {
        cout << "1" << endl; // Output 1 if string 'a' is lexicographically greater than 'b'.
        break;
      }
    }
  }
  return 0;
}
