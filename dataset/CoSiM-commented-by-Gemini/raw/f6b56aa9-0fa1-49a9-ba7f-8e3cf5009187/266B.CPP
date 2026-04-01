/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces (266B).
 * @details This program simulates the state changes in a queue of boys ('B') and girls ('G').
 * In each time step, any boy standing immediately in front of a girl swaps places.
 * This simulation is run for a specified number of time steps.
 *
 * Algorithm: Iterative single-pass swap simulation.
 * 1. Read the queue length `n`, time duration `t`, and the initial queue state string `s`.
 * 2. Loop `t` times to simulate each second.
 * 3. In each time step, perform a single pass over the string.
 * 4. During the pass, find adjacent 'B' and 'G' pairs and swap them.
 * 5. An important detail is skipping the next character after a swap to ensure a boy
 *    moves at most one position per time step.
 *
 * Time Complexity: O(T * N) where T is the time duration and N is the queue length.
 * Space Complexity: O(N) to store the queue string.
 */
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {

  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  /**
   * Block Logic: Orchestrates the temporal progression of the simulation.
   * Invariant: At the start of each iteration, `s` represents the queue state
   * after `t_initial - t` seconds have passed.
   */
  while (t > 0) {
    /**
     * Block Logic: Performs a single-pass sweep to resolve all possible swaps for one time step.
     * Logic: Identifies adjacent 'B' (Boy) and 'G' (Girl) pairs and swaps them
     * to simulate one step of movement.
     */
    for (int i = 0; i < n; i++)
      if (s[i] == 'B' && s[i + 1] == 'G') {
        char t = s[i];
        s[i] = s[i + 1];
        s[i + 1] = t;
        // Inline: Skips the next index to ensure that a boy who has just moved
        // does not move again in the same time step. This enforces atomic movement.
        i++;
      }
    t--;
  }
  // Output the final state of the queue after T seconds.
  cout << s;
}
