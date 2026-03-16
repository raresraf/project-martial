/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem (Codeforces 266B).
 *
 * @details
 * This program simulates the changes in a queue of boys ('B') and girls ('G').
 * The rule is that in each time step, any boy standing immediately in front of a
 * girl will swap places with her. The simulation runs for a given number of
 * time steps, `t`.
 *
 * The implementation consists of two nested loops. The outer loop iterates through
 * the time steps. The inner loop performs a single pass over the queue, swapping
 * every adjacent 'BG' pair it finds.
 *
 * A key detail is that after a swap at position `i`, the loop counter is advanced
 * by 2 (`i += 2`). This ensures that in a single pass (one time step), a child
 * moves at most one position, as the newly swapped 'B' at `i+1` is not considered
 * for another swap in the same pass.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * Time Complexity: O(t * n), where `t` is the number of time steps and `n` is the
 *                  length of the queue.
 * Space Complexity: O(n), to store the queue as a string.
 */
#include <iostream>
#include <string>    // For std::string
#include <utility>   // For std::swap

using namespace std;

int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;

  /**
   * @brief Outer loop to simulate the passage of time.
   * @details This loop runs for `t` iterations, where each iteration represents
   * one second of time.
   */
  for (int j = 0; j < t; ++j) {
    /**
     * @brief Inner loop to perform all swaps for the current time step.
     * @details This loop sweeps through the queue from left to right. When it finds
     * a 'B' followed by a 'G', it swaps them. The loop counter `i` is managed
     * manually within the loop body.
     */
    for (int i = 0; i < s.size() - 1;) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        // Inline: Advance the counter by 2. After swapping s[i] and s[i+1],
        // the next possible swap is at index i+1. This jump prevents a single
        // 'B' from moving multiple places in one time step.
        i += 2;
      } else {
        // If no swap occurs, simply move to the next position.
        ++i;
      }
    }
  }
  cout << s;
}
