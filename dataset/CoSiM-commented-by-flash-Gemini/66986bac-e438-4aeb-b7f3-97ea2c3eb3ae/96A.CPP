/**
 * @66986bac-e438-4aeb-b7f3-97ea2c3eb3ae/96A.CPP
 * @brief This program solves Codeforces problem 96A, "Football".
 * It analyzes a sequence of players (represented by '0's and '1's in a string)
 * to detect a "dangerous situation". A dangerous situation occurs if there are
 * 7 or more consecutive players of the same team (i.e., "0000000" or "1111111"
 * appears as a substring).
 *
 * Algorithm: String searching. Uses `std::string::find` to efficiently check
 * for the presence of predefined "dangerous" substrings.
 * Time Complexity: O(L * P) where L is the length of the input string and P
 * is the length of the pattern (7 in this case). Effectively O(L) as P is constant.
 * Space Complexity: O(1) beyond input string storage.
 */
#include <iostream> // For standard input/output operations (cin, cout).
#include <string>   // For using std::string and its methods like `find`.
using namespace std; // Uses the standard namespace.

/**
 * @brief Main function to detect a "dangerous situation" in a football game.
 * Reads a string representing player sequence and checks for 7 consecutive same-team players.
 * @return 0 on successful execution.
 */
int main() {
  string input; // Declares a string variable to store the input sequence of players.
  cin >> input; // Reads the player sequence from standard input.

  string subs1 = "0000000"; // Defines the substring representing 7 consecutive players of team '0'.
  string subs2 = "1111111"; // Defines the substring representing 7 consecutive players of team '1'.

  /**
   * Block Logic: Checks if either of the "dangerous situation" substrings ("0000000" or "1111111")
   * is present in the input string.
   * `string::npos` is a special value indicating that the substring was not found.
   * Invariant: Prints "YES" if a dangerous situation is found, "NO" otherwise.
   */
  if (input.find(subs1) != string::npos || input.find(subs2) != string::npos) // Checks if either `subs1` or `subs2` is found.
    cout << "YES"; // Prints "YES" if a dangerous situation exists.
  else
    cout << "NO"; // Prints "NO" if no dangerous situation is found.

  return 0; // Indicates successful program termination.
}