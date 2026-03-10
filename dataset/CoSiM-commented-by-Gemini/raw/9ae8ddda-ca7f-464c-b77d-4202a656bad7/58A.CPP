/**
 * @file 58A.CPP
 * @brief Solution for a competitive programming problem (Codeforces 58A - Chat Room).
 * The program checks if the word "hello" can be formed as a subsequence by
 * deleting zero or more characters from a given input string.
 *
 * @algorithm Greedy Linear Scan. The program iterates through the input string,
 * attempting to match the characters of "hello" in order.
 *
 * @time_complexity O(L), where L is the length of the input string, due to a single pass.
 * @space_complexity O(L) to store the input string.
 */
// CodeForces
// Problem 58A [Chat Room]

#include <bits/stdc++.h>
using namespace std;

int main() {
  string hello = "hello";

  string inputstr;
  cin >> inputstr;

  unsigned int strindex;
  // `checker` acts as a pointer to the current character being sought in "hello".
  unsigned int checker = 0;

  /**
   * @block
   * @brief Scans the input string to find the "hello" subsequence.
   *
   * Invariant: `checker` holds the index of the next character to find in the `hello` string.
   * The loop greedily matches characters. When the current character in `inputstr`
   * matches `hello[checker]`, `checker` is incremented to look for the next
   * character of "hello".
   */
  for (strindex = 0; strindex < inputstr.length(); strindex++) {
    if (inputstr[strindex] == hello[checker]) {
      checker++;
    }
  }

  // If `checker` is 5, it means all five characters of "hello" were found in order.
  if (checker >= 5) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}