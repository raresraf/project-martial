/**
 * @68cc0e5a-3ce0-4335-bdd0-55257824779e/96A.CPP
 * @brief This program solves Codeforces problem 96A, "Football".
 * It analyzes a sequence of players (represented by '0's and '1's in a string)
 * to detect a "dangerous situation". A dangerous situation occurs if there are
 * 7 or more consecutive players of the same team (i.e., "0000000" or "1111111"
 * appears as a substring).
 *
 * Algorithm: String searching using C-style `strstr` function. It checks for the
 * presence of two predefined "dangerous" substrings in the input string.
 * Time Complexity: O(L * P) in the worst case where L is the length of the input
 * string and P is the length of the pattern (7 in this case). Effectively O(L) as P is constant.
 * Space Complexity: O(L) for storing the input string.
 */
#include <bits/stdc++.h> // Includes many standard C++ library headers, including <iostream>, <string>, and <cstring> for strstr.

using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to detect a "dangerous situation" in a football game.
 * Reads a string representing player sequence and checks for 7 consecutive same-team players.
 * @return 0 on successful execution.
 */
int main() {
  string s; // Declares a string variable to store the input sequence of players.

  // Inline: Comment indicating the problem being solved.
  // checking substring problem

  cin >> s; // Reads the player sequence from standard input.

  string s1 = "0000000"; // Defines the substring representing 7 consecutive players of team '0'.
  string s2 = "1111111"; // Defines the substring representing 7 consecutive players of team '1'.

  /**
   * Block Logic: Checks if either of the "dangerous situation" substrings ("0000000" or "1111111")
   * is present in the input string `s`. The `strstr` function is used for this.
   * `s.c_str()` converts `std::string` to C-style string for `strstr`.
   * Invariant: Prints "YES" if a dangerous situation is found, "NO" otherwise.
   */
  if (strstr(s.c_str(), s1.c_str())) // Checks if `s1` is a substring of `s`.
    cout << "YES" << endl;           // Prints "YES" if `s1` is found.
  else if (strstr(s.c_str(), s2.c_str())) // Checks if `s2` is a substring of `s` (only if `s1` was not found).
    cout << "YES" << endl;           // Prints "YES" if `s2` is found.
  else
    cout << "NO" << endl; // Prints "NO" if neither `s1` nor `s2` is found.

  return 0; // Indicates successful program termination.
}