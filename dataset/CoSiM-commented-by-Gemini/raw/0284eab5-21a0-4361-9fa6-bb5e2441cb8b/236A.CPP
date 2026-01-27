/**
 * @file 236A.CPP
 * @brief Solution for Codeforces problem 236A "Boy or Girl".
 *
 * This program determines if a username is "boy" or "girl" based on the parity
 * of the number of unique characters in the username. If the number of unique
 * characters is odd, it's "IGNORE HIM!", otherwise it's "CHAT WITH HER!".
 *
 * Algorithm: Count unique characters using a hash set, then check parity.
 * Time Complexity: O(L) where L is the length of the input string.
 * Space Complexity: O(C) where C is the size of the character set (e.g., 26 for lowercase English letters).
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // Functional Utility: Optimizes C++ standard streams for faster input/output operations.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);
  string in;
  // Block Logic: Reads the input username as a string.
  cin >> in;
  // Block Logic: Creates an unordered_set to store unique characters from the input string.
  // The unordered_set automatically handles uniqueness.
  unordered_set<char> s(in.begin(), in.end());
  /**
   * Block Logic: Determines the output based on the parity of unique characters.
   * If the number of unique characters is odd, output "IGNORE HIM!".
   * Otherwise (even number of unique characters), output "CHAT WITH HER!".
   * Invariant: `s.size()` holds the count of unique characters.
   */
  if (s.size() % 2)
    cout << "IGNORE HIM!";
  else
    cout << "CHAT WITH HER!";
  return 0;
}
