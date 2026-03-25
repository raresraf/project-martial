#include <algorithm>
#include <iostream>
#include <string>

/**
 * @file 96A.CPP
 * @brief Solution for a string processing problem, often called "Football".
 *
 * Problem Statement:
 * Given a string representing a sequence of players from two teams (marked '0'
 * and '1'), determine if the situation is "dangerous". A situation is
 * dangerous if there are 7 or more consecutive players from the same team.
 *
 * Intended Algorithm:
 * The code attempts to solve this by iterating through the string and keeping
 * a running count of consecutive identical characters.
 *
 * 1. Read the input string.
 * 2. Initialize counters for consecutive '0's and '1's.
 * 3. Loop through the string, comparing adjacent characters.
 * 4. If two adjacent characters are the same, increment the corresponding counter.
 * 5. If they are different, reset the counters.
 * 6. After the loop, check if any of the recorded maximum counts are 7 or more.
 *
 * @note The implementation has a logical flaw. The variables `myone` and
 * `myzero` are only updated when a consecutive sequence is extended. They do
 * not necessarily store the maximum sequence length found throughout the entire
 * string. Furthermore, if the string has no consecutive identical characters
 * (e.g., "101010"), these variables might be used uninitialized, leading to
 * undefined behavior. A more robust solution would be to use `std::string::find`
 * to search for the substrings "0000000" or "1111111".
 *
 * Time Complexity: O(L), where L is the length of the string.
 * Space Complexity: O(L) to store the input string.
 */

using namespace std;

int main() {
  string s;
  cin >> s;

  // Counters for the current consecutive runs of '1's and '0's.
  int count = 1;
  int zcount = 1;

  // Variables to store the maximum length of consecutive runs found.
  // These are potentially used uninitialized if no consecutive characters are found.
  int myone = 0, myzero = 0;
  if (s.length() > 0) {
      myone = 1;
      myzero = 1;
  }

  // Block Logic: Iterate through the string to find consecutive characters.
  for (int i = 0; i < s.length() - 1; i++) {
    // Pre-condition: Check for a pair of consecutive '1's.
    if (s[i] == '1' && s[i] == s[i + 1]) {
      count++;
      myone = max(myone, count); // Update the max if current run is longer.
    }
    // Pre-condition: Check for a pair of consecutive '0's.
    else if (s[i] == '0' && s[i] == s[i + 1]) {
      zcount++;
      myzero = max(myzero, zcount); // Update the max if current run is longer.
    } else {
      // If the sequence is broken, reset the running counters.
      count = 1;
      zcount = 1;
    }
  }

  // Final check: Determine if any run of 7 or more was found.
  if (myone >= 7 || myzero >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}