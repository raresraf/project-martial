
#include <bits/stdc++.h>
using namespace std;

/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A, "Football".
 * @details The program determines if a situation is "dangerous" in a football
 * match, which is defined as having at least seven consecutive players of the
 * same team.
 *
 * Algorithm:
 * The program iterates through the input string, which represents a line of
 * players. It maintains a counter for the current number of consecutive
 * identical characters ('0' or '1'). The counter is incremented for each
 * identical adjacent character. If a different character is encountered, the
 * counter is reset to 1, but only if it hasn't already reached the dangerous
 * threshold of 7. This ensures that once a dangerous situation is found, the
 * state is "latched". After the loop, a final check confirms if the counter
 * reached 7 at any point.
 *
 * Time Complexity: O(N), where N is the length of the input string, due to a
 * single pass over the string.
 * Space Complexity: O(N) to store the input string.
 */
int main() {
  string situation;
  cin >> situation;

  // Optimization: A dangerous situation is impossible if there are fewer than 7 players.
  if (situation.size() < 7) {
    cout << "NO";
  } else {
    // `counter` tracks the number of consecutive identical players.
    int counter = 1;
    /**
     * @brief Iterate through the line of players to find a dangerous situation.
     *
     * Invariant: `counter` holds the length of the current consecutive run
     * ending at position `i-1`.
     */
    for (int i = 1; i < situation.size(); i++) {
      if (situation[i] == situation[i - 1]) {
        counter += 1;
      } else if (counter < 7) {
        // If the run is broken and we haven't found a dangerous situation yet,
        // reset the counter for the new run. If `counter` is already >= 7,
        // this condition is false, effectively "latching" the positive result.
        counter = 1;
      }
    }

    // Final check after the loop.
    if (counter >= 7) {
      cout << "YES";
    } else {
      cout << "NO";
    }
  }

  return 0;
}
