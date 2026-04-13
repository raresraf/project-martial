/**
 * @file 339A.CPP
 * @brief This program parses an input string representing a sum of single-digit numbers
 *        (e.g., "1+2+3"). It extracts the numerical digits, sorts them in
 *        non-decreasing order, and then reconstructs the sum string with digits
 *        separated by '+' signs.
 * Algorithm: String parsing, Sorting, String reconstruction.
 * Time Complexity: O(N log N) where N is the count of digits. Sorting dominates,
 *                  while digit extraction and string reconstruction are O(N).
 * Space Complexity: O(N) for storing digits in a vector and building the output string.
 */
/*
    Problem Link    :   https://codeforces.com/problemset/problem/339/A
    Contest Link    :   https://codeforces.com/contest/339/problem/A
    Status          :   Accepted

    Author          :   Nikhil Nagrale
    Codeforces      :   https://codeforces.com/profile/nikhilnagrale2
    Codechef        :   https://www.codechef.com/users/nikhilnagrale2
    Github          :   https://github.com/nikhilnagrale2

    Copyright       :   Solve it on your own.
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
  string s; // Input string containing the sum.
  // Reads the input sum string.
  cin >> s;
  vector<char> v; // Vector to store extracted digit characters.
  /**
   * Block Logic: Iterates through the input string to extract only the digit characters.
   * Invariant: Each character `x` from `s` is checked; if it's not a '+', it's added to vector `v`.
   */
  for (char x : s) {
    if (x != '+') // Checks if the character is not a '+' sign.
      v.push_back(x); // Adds the digit character to the vector.
  }
  // Sorts the extracted digit characters in non-decreasing order.
  sort(v.begin(), v.end());
  string ans = ""; // Initializes an empty string to build the sorted sum.
  ans += v[0];     // Appends the first (smallest) digit to the answer string.
  // Removes the first digit from the vector as it has been added to 'ans'.
  // This is potentially inefficient for large vectors but fine for small N.
  v.erase(v.begin());

  /**
   * Block Logic: Appends the remaining sorted digits to the answer string,
   *              each preceded by a '+' sign.
   * Invariant: For each remaining digit `x` in `v`, a '+' and then `x` are appended to `ans`.
   */
  for (char x : v) {
    ans += '+'; // Appends a '+' character.
    ans += x;   // Appends the current digit character.
  }
  // Prints the final sorted sum string.
  cout << ans << endl;
  return 0;
}