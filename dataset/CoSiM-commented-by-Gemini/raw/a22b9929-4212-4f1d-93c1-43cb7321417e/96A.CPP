/**
 * @file 96A.CPP
 * @brief Solution to a competitive programming problem that checks for a consecutive sequence of 7 identical characters.
 *
 * This program determines if a given string, representing a sequence of football players, 
 * contains a "dangerous" situation. A dangerous situation is defined as having at least
 * seven consecutive players of the same team (represented by '0' or '1').
 *
 * @algorithm Linear Scan
 * @time_complexity O(N), where N is the length of the input string. The program iterates through the string once.
 * @space_complexity O(N) for storing the input string.
 */
#include <iostream>
using namespace std;

int main() {
  string players;
  cin >> players;
  int zeros, ones;
  string result = "NO";
  zeros = ones = 0;
  /**
   * @brief Iterates through the string of players to find consecutive sequences.
   *
   * This loop scans the input string character by character, keeping a running
   * count of consecutive '0's and '1's. The counters are reset whenever the
   * character changes.
   */
  for (int i = 0; i < players.length(); i++) {
    if (players[i] == '1') {
      ones++;
      zeros = 0;
    }
    if (players[i] == '0') {
      zeros++;
      ones = 0;
    }
    /**
     * @brief Checks for a dangerous situation.
     *
     * If a sequence of 7 consecutive '0's or '1's is found, the result
     * is updated to "YES", and the loop could be terminated for optimization,
     * but isn't in this implementation.
     */
    if ((zeros == 7) || (ones == 7))
      result = "YES";
  }
  cout << result << "\n";
}