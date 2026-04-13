/**
 * @file 96A.CPP
 * @brief This program solves the "Football" problem, detecting if a situation is "dangerous".
 * A situation is considered dangerous if there are 7 or more consecutive players of the same team.
 * Players are represented by '0' or '1' in an input string.
 * Algorithm: Iterative counting of consecutive identical characters with reset on change.
 * Note: The implementation of `myone` and `myzero` is designed to store the count of the
 * *last* consecutive sequence of '1's or '0's encountered, which may not correctly
 * capture the maximum consecutive count if the longest sequence is in the middle of the string.
 * Time Complexity: O(N), where N is the length of the input string, due to a single pass.
 * Space Complexity: O(N) for storing the input string.
 */
#include <algorithm> // Included in original, not explicitly used for this logic.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for string manipulation.

using namespace std;

int main() {
  string s; // Stores the input string representing the sequence of players.
  // Reads the input string from standard input.
  cin >> s;

  int count = 1;  // Counts consecutive '1's. Initializes to 1 for the first character.
  int zcount = 1; // Counts consecutive '0's. Initializes to 1 for the first character.
  int myone = 0;  // Stores the last recorded consecutive count of '1's.
  int myzero = 0; // Stores the last recorded consecutive count of '0's.
  
  /**
   * Block Logic: Iterates through the player string to find sequences of identical characters.
   * It maintains a count of consecutive '1's ('count') and '0's ('zcount').
   * If a sequence of '1's continues, 'count' increments. If '0's continue, 'zcount' increments.
   * If the sequence breaks, the respective counter is reset to 1.
   * The variables `myone` and `myzero` are updated with `count` or `zcount` only when a consecutive
   * sequence of '1's or '0's is being extended. This means they will hold the count of the last
   * consecutive sequence ending at that point, which might not be the overall maximum.
   * Invariant: `count` and `zcount` reflect the current consecutive run lengths.
   */
  for (int i = 0; i < s.length() - 1; i++) {
    // If the current and next characters are '1', increment 'count' and update 'myone'.
    if (s[i] == '1' && s[i] == s[i + 1]) {
      count++;
      myone = count;
    }
    // Else if the current and next characters are '0', increment 'zcount' and update 'myzero'.
    else if (s[i] == '0' && s[i] == s[i + 1]) {
      zcount++;
      myzero = zcount;
    }
    // If the sequence breaks (e.g., '1' followed by '0', or '0' followed by '1'), reset both counters.
    else {
      count = 1;
      zcount = 1;
    }
  }
  /**
   * Conditional Logic: After iterating through the entire string, determines if a "dangerous" situation occurred.
   * This check relies on the final values of `myone` and `myzero`. If either of these
   * (representing the count of the last consecutive sequence) is 7 or more, it outputs "YES".
   * Otherwise, it outputs "NO".
   */
  if (myone >= 7 || myzero >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  // Standard program exit.
  return 0;
}
