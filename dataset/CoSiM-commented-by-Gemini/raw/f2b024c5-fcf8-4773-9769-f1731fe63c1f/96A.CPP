/**
 * @file 96A.CPP
 * @brief Determines if a football situation is "dangerous".
 *
 * This program solves the "Football" problem, where a situation is considered
 * "dangerous" if there are at least seven consecutive players from the same team.
 * The teams are represented by '0' and '1'. The program reads a string of '0's
 * and '1's representing the players and checks if it contains a substring of
 * seven identical characters.
 *
 * @author f2b024c5-fcf8-4773-9769-f1731fe63c1f
 *
 * @algorithm Substring Search
 * @complexity
 * Time: O(N), where N is the length of the input string. The `string::find`
 *       operation has a time complexity that is, in the worst case, proportional
 *       to the length of the string.
 * Space: O(N), to store the input string and its copy.
 */
#include <iostream>
#include <string> // Included for std::string and std::string::npos

using namespace std;

int main() {
  // `players`: stores the input line of players.
  string players = "";
  // `chance`: an unnecessary copy of the `players` string.
  string chance = "";
  // `result`: stores the final answer "YES" or "NO".
  string result = "";
  // `s1` and `s2`: the two "dangerous" patterns to search for.
  string s1 = "1111111";
  string s2 = "0000000";

  // Read the player formation from standard input.
  cin >> players;

  /**
   * @brief This loop redundantly copies the `players` string to `chance`.
   * The subsequent logic could have used `players` directly.
   */
  for (int i = 0; i < players.length(); i++) {
    chance += players[i];
  }

  /**
   * @brief Check for the presence of a dangerous situation.
   * `string::find` searches for the first occurrence of a substring.
   * If the substring is not found, it returns `string::npos`.
   */
  if (chance.find(s2) != string::npos) { // Check for seven consecutive '0's.
    result = "YES";
  } else if (chance.find(s1) != string::npos) { // Check for seven consecutive '1's.
    result = "YES";
  } else {
    // If neither pattern is found, the situation is not dangerous.
    result = "NO";
  }

  // Print the final result.
  cout << result << endl;
  return 0;
}