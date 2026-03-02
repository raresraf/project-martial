#include <iostream>
#include <string>
using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to the "Football" problem from Codeforces (Problem 96A).
 * @details The program determines if a football situation is "dangerous", which occurs
 * if there are at least seven consecutive players from the same team.
 *
 * Algorithm:
 * The program reads a string of '0's and '1's, representing players of two different teams.
 * It then uses the `string::find` method to search for the substrings "0000000" and "1111111".
 * The presence of either substring indicates that at least seven players of one team are
 * positioned consecutively. If found, the program outputs "YES"; otherwise, it outputs "NO".
 *
 * Time Complexity: O(L), where L is the length of the input string. This is the complexity
 * of the `string::find` operation.
 * Space Complexity: O(L) to store the input string. The space for the target substrings
 * is constant.
 */
int main() {
  string input;
  // Read the string representing the players' configuration.
  cin >> input;

  // Define the two "dangerous" patterns.
  string subs1 = "0000000";
  string subs2 = "1111111";

  /**
   * @brief Check for the existence of a dangerous situation.
   * `string::find` returns the starting position of the substring if found,
   * and `string::npos` if not found. The condition checks if either of the
   * two dangerous patterns exists in the input string.
   */
  if (input.find(subs1) != string::npos || input.find(subs2) != string::npos)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}