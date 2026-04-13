/**
 * @file 96A.CPP
 * @brief This program determines if a given sequence of players contains a "dangerous" situation.
 * A dangerous situation is defined as 7 consecutive players of the same type (either '0' or '1').
 * Algorithm: String searching for predefined dangerous patterns.
 * Time Complexity: O(N * M) where N is the length of the players string and M is the length of the dangerous pattern (7).
 * Space Complexity: O(N) for storing the input and temporary strings.
 */
#include <iostream>

using namespace std;

int main() {
  string players = "";
  string chance = "";
  string result = "";
  string s1 = "1111111";
  string s2 = "0000000";
  cin >> players;
  /**
   * Block Logic: Copies the input 'players' string into 'chance'.
   * This loop essentially duplicates the 'players' string.
   */
  for (int i = 0; i < players.length(); i++) {
    chance += players[i];
  }
  /**
   * Block Logic: Checks for the presence of dangerous sequences.
   * A dangerous sequence is 7 consecutive '0's or '1's.
   * If either sequence is found, the result is "YES", otherwise "NO".
   * Invariant: 's1' and 's2' represent the dangerous patterns to be matched.
   */
  if (chance.find(s2) != string::npos) {
    result = "YES";
  } else if (chance.find(s1) != string::npos)

  {
    result = "YES";
  } else {
    result = "NO";
  }

  // Output the determination of whether a dangerous situation exists ("YES" or "NO").
  cout << result << endl;
  return 0;
}

