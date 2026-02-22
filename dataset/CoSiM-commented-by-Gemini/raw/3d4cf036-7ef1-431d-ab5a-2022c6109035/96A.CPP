/**
 * @file 96A.CPP
 * @brief Solution for the "Football" problem (Codeforces 96A).
 *
 * @algorithm
 * The problem is to determine if there are at least 7 consecutive players of the
 * same team. The team is represented by '0' or '1'.
 *
 * This solution uses a state machine-like approach to find consecutive players
 * in a single pass:
 * 1. Read the entire line of players into a string.
 * 2. Initialize a `state` variable to track the current team ('0' or '1')
 *    in a sequence.
 * 3. Initialize `runningSum` to count the length of the current sequence.
 * 4. Iterate through the string character by character.
 * 5. If the current character is different from the `state`, it means a new
 *    sequence has started. Reset `runningSum` and update `state` to the
 *    new character.
 * 6. If the current character is the same as `state`, just increment `runningSum`.
 * 7. After each increment, check if `runningSum` has reached 7. If it has, the
 *    condition is met, the answer is "YES", and we can terminate early.
 * 8. If the loop completes without the condition being met, the answer is "NO".
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the input string, as we
 *   do a single pass over the string.
 * - Space Complexity: O(N) to store the input string.
 */
#include <iostream>
// Note: Including <string> is good practice for std::string and getline.
using namespace std;

int main() {
  const int dangerous = 7; // The minimum number of consecutive players for a dangerous situation.
  string line;
  getline(cin, line);

  int state = -1;       // Tracks the current player in the sequence ('0' or '1'). Initialized to a sentinel value.
  int runningSum = 0;   // Counts the length of the current consecutive sequence.
  string output = "NO"; // The final answer, defaults to "NO".
  /**
   * @brief Main loop to iterate through the string of players.
   */
  for (int k = 0; k < line.size(); k++) {
    /**
     * @brief Check if the current player breaks the consecutive sequence.
     * If the player `line[k]` is different from the current `state`,
     * a new sequence begins.
     */
    if (line[k] != state) {
      state = line[k];
      runningSum = 0; // Reset the counter for the new sequence.
    }
    // Increment the counter for the current player sequence.
    ++runningSum;
    /**
     * @brief If the sequence reaches the dangerous threshold, set the output
     * and exit the loop as the condition has been met.
     */
    if (runningSum >= dangerous) {
      output = "YES";
      break;
    }
  }
  cout << output << endl;
}
