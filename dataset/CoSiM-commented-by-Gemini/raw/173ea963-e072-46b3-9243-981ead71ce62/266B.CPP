/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem using ASCII value comparison.
 * @details This program simulates a queue of children ('B' for boy, 'G' for girl).
 * For a given time `t`, any boy standing immediately in front of a girl will swap
 * places with her. This process is repeated for `t` seconds.
 *
 * @section algorithm Algorithm
 * The program performs a direct simulation of the process using a clever trick for comparison.
 * 1. Read the number of children `n`, the total time `t`, and the initial queue string `s`.
 * 2. An outer loop runs `t` times, each iteration representing one second of simulation time.
 * 3. An inner loop iterates through the queue from left to right to perform all possible swaps for that second.
 *    - It checks for an adjacent "BG" pair by comparing their ASCII values. Since the
 *      ASCII value of 'B' (66) is less than 'G' (71), the condition `(int)str[j] < (int)str[j+1]`
 *      is a compact way of writing `str[j] == 'B' && str[j+1] == 'G'`.
 *    - When such a pair is found, the characters are swapped.
 *    - After a swap at index `j`, the loop index `j` is advanced by an extra 1
 *      (for a total increment of 2 for that step, due to the loop's own increment). This is
 *      a key step to ensure a child does not move more than one position per second.
 * 4. After `t` seconds, the final state of the queue is printed, preceded by a newline.
 *
 * @section complexity Complexity Analysis
 * - Time Complexity: O(t * n), where `t` is the time and `n` is the length of the queue, due to the nested loops.
 * - Space Complexity: O(n) to store the queue string.
 */
#include <iostream>

using namespace std;

int main() {

  string str;
  int n;
  int t;
  char tmp;

  cin >> n >> t;
  cin.ignore(); // Consume the newline character left in the input buffer by cin.
  getline(cin, str);

  // The outer loop simulates the passage of 't' seconds.
  for (int i = 1; i <= t; ++i) {

    // The inner loop performs one pass of swaps over the queue for the current second.
    for (int j = 0; j < n - 1; ++j) {
      // This condition cleverly checks for a 'B' followed by a 'G'
      // by comparing their ASCII values, since ASCII('B') < ASCII('G').
      if ((int)str[j] < (int)str[j + 1]) {
        // If a "BG" pair is found, swap them.
        tmp = str[j];
        str[j] = str[j + 1];
        str[j + 1] = tmp;
        // Increment j by an additional 1 to skip the next position. This ensures
        // that the boy who just moved does not move again in the same time step.
        j += 1;
      }
    }
  }

  // Print a newline, then the final state of the queue.
  cout << endl << str;

  return 0;
}
