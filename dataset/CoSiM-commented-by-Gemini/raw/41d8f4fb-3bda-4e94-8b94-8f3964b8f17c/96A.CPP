/**
 * @file 96A.CPP
 * @brief Solution to Codeforces problem 96A - Football.
 *
 * @algorithm
 * The problem defines a "dangerous" situation as having at least 7 consecutive
 * players from the same team. The input is a string of '0's and '1's. This
 * program checks if such a consecutive sequence exists.
 *
 * The algorithm works by iterating through the string and counting consecutive
 * identical characters:
 * 1. Initialize a `runningSum` of consecutive players to 0 and a `state`
 *    variable to track the current player type ('0' or '1').
 * 2. Loop through each character of the input string.
 * 3. If the current character is different from the tracked `state`, a new
 *    sequence has begun. Reset the `runningSum` to 0 and update the `state`
 *    to the new character.
 * 4. Increment the `runningSum` for the current sequence.
 * 5. If `runningSum` ever reaches 7, the condition is met. Set the output to "YES"
 *    and break the loop immediately.
 * 6. If the loop completes without the condition being met, the output remains "NO".
 * 7. Print the final result.
 *
 * @complexity
 * - Time Complexity: O(N), where N is the length of the input string, as it
 *   requires only a single pass through the string.
 * - Space Complexity: O(N) to store the input string.
 */
#include <iostream>
using namespace std;

int main() {
  const int dangerous = 7;
  string line;
  getline(cin, line);
  int state = -1; // Using -1 as an initial state different from '0' or '1'.
  int runningSum = 0;
  string output = "NO"; // Default output is "NO".
  /**
   * @brief Iterates through the string to find a sequence of 7 identical characters.
   */
  for (int k = 0; k < line.size(); k++) {
    // Check if the current character breaks the consecutive sequence.
    if (line[k] != state) {
      // If so, reset the counter and update the state to the new character.
      state = line[k];
      runningSum = 0;
    }
    // Increment the count for the current consecutive sequence.
    ++runningSum;
    // If the count reaches the dangerous threshold, set the result and exit.
    if (runningSum >= dangerous) {
      output = "YES";
      break;
    }
  }
  cout << output << endl;
}
