/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - "Football".
 * @details The program determines if a "dangerous" situation has occurred, which is defined
 *          as having at least seven consecutive players of the same team ('0' or '1').
 *          This implementation uses an efficient single-pass approach.
 *
 * Algorithm:
 * 1. Read the input string.
 * 2. Initialize a counter for consecutive identical characters to 1.
 * 3. Iterate through the string from the second character.
 * 4. If the current character is the same as the previous one, increment the counter.
 *    a. If the counter reaches 7, a dangerous situation is found. Print "YES" and exit.
 * 5. If the current character is different from the previous one, reset the counter to 1.
 * 6. If the loop finishes without the counter reaching 7, print "NO".
 *
 * Time Complexity: O(N), where N is the length of the string, due to the single pass.
 * Space Complexity: O(N) for storing the input string.
 */
// 96A.cpp
// A. Football
// Author: Ikki
// May 31, 2015

#include <iostream>
#include <string>
using namespace std;

int main() {
  string str;
  cin >> str;
  // `cnt` tracks the number of consecutive identical characters.
  int cnt = 1;
  /**
   * @brief Main loop to iterate through the string and count consecutive players.
   * Invariant: `cnt` holds the length of the current run of identical characters ending at index `i-1`.
   */
  for (int i = 1; i < str.size(); i++) {
    // Check if the current character continues the sequence.
    if (str[i] == str[i - 1]) {
      cnt++;
      /**
       * @brief Check if a dangerous situation has occurred.
       * If the count of consecutive players reaches 7, the condition is met.
       */
      if (cnt >= 7) {
        cout << "YES" << endl;
        return 0;
      }
    } else {
      // If the sequence is broken, reset the counter.
      cnt = 1;
    }
  }
  // If the loop completes, no sequence of 7 was found.
  cout << "NO" << endl;
  return 0;
}
