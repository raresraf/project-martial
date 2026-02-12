/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 * @details The problem defines a "dangerous" situation in a football match if there are at least
 * seven consecutive players of the same team. The players are represented by a string of '0's and '1's.
 * This program checks the input string for any occurrence of 7 consecutive identical characters.
 *
 * @algorithm Sliding window check for consecutive characters.
 * @complexity
 * Time complexity: O(N), where N is the length of the input string, as it iterates through the string once.
 * Space complexity: O(1), besides the storage for the input string itself.
 */
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

/**
 * @brief Checks if the given player string represents a dangerous situation.
 * @param s The string of '0's and '1's representing players.
 * @return "YES" if there are 7 or more consecutive identical players, "NO" otherwise.
 */
string solve(string s) {
  // `counter` tracks the number of consecutive pairs of identical players.
  // A count of 6 pairs means 7 consecutive players.
  int counter = 0;

  /**
   * @brief Iterates through the string to find consecutive players.
   * @invariant `counter` holds the number of identical pairs ending at the current position.
   */
  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] == s[i + 1]) {
      counter++;

      // If we have found 6 consecutive pairs, that means 7 players in a row.
      if (counter == 6) {
        return "YES";
      }

      continue; // Continue to the next character pair.
    }
    // If the characters are different, reset the counter.
    counter = 0;
  }
  
  // If the loop completes without finding 7 consecutive players.
  return "NO";
}

/**
 * @brief A set of test cases to verify the `solve` function's correctness.
 */
void test() {
  assert(solve("1000000001") == "YES");
  assert(solve("00100110111111101 ") == "YES");
  assert(solve("001001") == "NO");
  assert(solve("11110111011101 ") == "NO");
}

int main() {
  // For reading from a local file during testing.
  // freopen( "input.txt", "r", stdin );

  string s;
  cin >> s;

  // The test function is commented out for submission.
  // test();

  cout << solve(s) << endl;
}