/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - Football.
 * @description This program determines if a situation is "dangerous" in a football match.
 * A situation is considered dangerous if there are at least 7 consecutive players of the same team.
 * The input is a string of '0's and '1's, each digit representing a player from one of the two teams.
 *
 * Algorithm: The program iterates through the input string, keeping a running count of
 * consecutive identical characters. If a character is the same as the previous one, the count
 * is incremented. If it's different, the count is reset to 1. If the count ever reaches 7,
 * the situation is dangerous, and the program can stop early.
 *
 * Time Complexity: O(N), where N is the length of the input string, because it involves a single pass.
 * Space Complexity: O(N) to store the input string.
 */
#include <iostream>
#include <string>
using namespace std;
int main() {
  // a: the string representing the sequence of players.
  string a;
  // c: a counter for consecutive identical players.
  int c;
  cin >> a;
  c = 1;

  /**
   * Block Logic: Iterate through the string to find consecutive players.
   * Invariant: 'c' holds the number of consecutive identical players ending at index 'i'.
   * The loop compares each character `a[j]` with the one before it `a[i]`.
   */
  for (int i = 0, j = 1; j < a.length(); j++, i++) {
    // If the current player is from the same team as the previous one, increment the counter.
    if (a[i] == a[j])
      ++c;
    else
      // If the team changes, reset the counter.
      c = 1;
    
    // If a sequence of 7 has been found, the condition is met.
    if (c == 7)
      break;
  }

  // Post-condition: Check if a dangerous situation was detected.
  if (c == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
