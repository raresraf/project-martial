/**
 * @file 236A.CPP
 * @brief Solution to a competitive programming problem: "Boy or Girl".
 *
 * This program determines whether a person should "CHAT WITH HER!" or "IGNORE HIM!"
 * based on the distinct characters in their username. If the number of distinct
 * characters in the username is even, the output is "CHAT WITH HER!", otherwise
 * it's "IGNORE HIM!".
 *
 * Algorithm: Count distinct characters in a string using a set, then check parity.
 * Time Complexity: O(L log K), where L is the length of the username string and K
 *                  is the number of distinct characters (K <= 26 for English alphabet),
 *                  due to populating the set.
 */
#include <bits/stdc++.h> // Functional Utility: Includes most standard library headers,
                         // including <iostream>, <string>, <set>, <cstring> for competitive programming.
using namespace std;     // Functional Utility: Uses the standard namespace to avoid prefixing standard library elements with std::

int main() {
  // Functional Utility: Optimizes standard I/O streams for faster execution,
  // crucial in competitive programming for time-sensitive tasks.
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  // Variable Declarations:
  // s: Character array to store the input username.
  char s[100];
  // Input: Reads the username string.
  cin >> s;

  // Block Logic: Creates a set of characters from the input string to count distinct characters.
  // std::set automatically stores only unique elements and keeps them sorted.
  // The range [s, s + strlen(s)) covers all characters in the C-style string 's'.
  set<char> s1(s, s + strlen(s));

  // Block Logic: Checks the parity of the number of distinct characters.
  // If the count of distinct characters is even, it prints "CHAT WITH HER!".
  if (s1.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else // If the count is odd, it prints "IGNORE HIM!".
    cout << "IGNORE HIM!";

  return 0; // Functional Utility: Indicates successful program execution.
}
