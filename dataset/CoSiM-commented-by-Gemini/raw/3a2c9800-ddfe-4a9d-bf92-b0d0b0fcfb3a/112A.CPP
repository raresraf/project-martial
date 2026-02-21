/**
 * @file 112A.CPP
 * @brief A C++ solution to the "Petya and Strings" problem (Codeforces 112A).
 *
 * @details This program performs a case-insensitive lexicographical comparison of two
 * strings. It determines if the first string is less than, greater than, or
 * equal to the second string, ignoring differences in case.
 *
 * @algorithm The approach is to normalize the case of both strings before comparison.
 * 1. The `std::transform` algorithm is used to convert both input strings to
 *    uppercase in-place.
 * 2. A loop then iterates through the now-uppercase strings, comparing them
 *    character by character.
 * 3. The result is set to -1 or 1 as soon as the first lexicographical
 *    difference is found. If the loop completes without finding any
 *    difference, the strings are equal, and the result remains 0.
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the strings. This includes O(L)
 *                  for the two `transform` calls and O(L) for the comparison loop.
 * Space Complexity: O(L), to store the two input strings.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Fast I/O setup.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  string a, b;
  cin >> a >> b;
  
  // Pre-computation: Convert both strings to uppercase for case-insensitive comparison.
  transform(a.begin(), a.end(), a.begin(), ::toupper);
  transform(b.begin(), b.end(), b.begin(), ::toupper);

  int x = 0; // The result: -1, 0, or 1.
  /**
   * @brief Block Logic: Iterates through the normalized strings to find the
   * first lexicographical difference.
   */
  for (int i = 0; i < a.length(); i++) {
    if ((int)a[i] > (int)b[i]) {
      x = 1;
      break;
    } else if ((int)a[i] < (int)b[i]) {
      x = -1;
      break;
    }
  }
  cout << x << endl;
}