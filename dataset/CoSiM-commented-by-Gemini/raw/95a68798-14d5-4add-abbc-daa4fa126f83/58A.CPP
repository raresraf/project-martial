/**
 * @file 58A.CPP
 * @brief Checks if the word "hello" is a subsequence of an input string.
 *
 * This program solves the "Chat room" problem from Codeforces. It determines if
 * it's possible to form the word "hello" by deleting zero or more characters from a
 * given string. This is a classic subsequence problem.
 *
 * @algorithm State Machine / Greedy Subsequence Matching
 * @complexity
 * - Time: O(L), where L is the length of the input string, because it iterates
 *   through the input string once. String comparisons inside the loop take
 *   constant time as `s` has a max length of 5.
 * - Space: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

int main() {
  // I/O optimizations for speed.
  ios::sync_with_stdio(false);
  cin.tie(NULL);
  cout.tie(NULL);

  string str;
  cin >> str;

  // `s` will be built up to "hello" as its characters are found in order.
  string s;

  /**
   * @brief Block Logic: Implements a state machine to find the "hello" subsequence.
   * This loop iterates through the input string `str`, looking for the characters
   * of "hello" in the correct sequence.
   */
  for (int i = 0; i < str.size(); i++) {
    // State 0: Looking for 'h'.
    if (s.empty() && str[i] == 'h')
      s.push_back('h');
    // State 1: Found 'h', looking for 'e'.
    else if (s == "h" && str[i] == 'e')
      s.push_back('e');
    // State 2: Found "he", looking for the first 'l'.
    else if (s == "he" && str[i] == 'l')
      s.push_back('l');
    // State 3: Found "hel", looking for the second 'l'.
    else if (s == "hel" && str[i] == 'l')
      s.push_back('l');
    // State 4: Found "hell", looking for 'o'.
    else if (s == "hell" && str[i] == 'o')
      s.push_back('o');
  }

  /**
   * @brief Block Logic: Final check to see if the full subsequence was found.
   */
  (s == ("hello")) ? cout << "YES\n" : cout << "NO\n";
}
