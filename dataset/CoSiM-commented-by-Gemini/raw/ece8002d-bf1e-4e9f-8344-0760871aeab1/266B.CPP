/**
 * @file 266B.CPP
 * @brief Simulates a queue of boys and girls swapping positions over time.
 * @note This is a different implementation for the same problem as seen in other files.
 *       It uses C-style I/O and a `vector<bool>` to represent the queue state.
 *
 * This program models a discrete-time process where adjacent pairs of a boy ('B')
 * followed by a girl ('G') swap their positions. This is repeated for a specified
 * number of time steps.
 *
 * @b Algorithm: Direct simulation of local swaps using a boolean vector where 1 represents
 *    a boy and 0 represents a girl.
 * @b Time_Complexity: O(T * N), where T is the number of time steps and N is the queue length.
 * @b Space_Complexity: O(N) to store the queue state.
 */
// https://codeforces.com/problemset/problem/266/B
// SOLUTION BY DN54321

#include <algorithm>
#include <ctype.h>
#include <functional>
#include <iostream>
#include <string.h>
#include <vector>

typedef long long ll;
using namespace std;

int main() {
  int n, t;
  char g;
  scanf("%d%d", &n, &t);
  vector<bool> v(n);
  getchar(); // Consume the newline character left by scanf.
  
  // Read the initial state of the queue into a boolean vector.
  for (int i = 0; i < n; ++i) {
    scanf("%c", &g);
    if (g == 'B')
      v[i] = 1; // 1 represents a Boy.
    else
      v[i] = 0; // 0 represents a Girl.
  }

  /**
   * @brief Main simulation loop that advances time for 't' steps.
   */
  for (int i = 0; i < t; ++i) {
    /**
     * @brief A single-pass scan through the queue to perform swaps for the current time step.
     */
    for (int j = 0; j < n - 1; ++j) {
      // Check for a Boy followed by a Girl (1 followed by 0).
      if (v[j] && !v[j + 1]) {
        // Perform the swap.
        v[j] = 0;
        v[j + 1] = 1;
        /**
         * @brief Advance the index to prevent the newly moved Boy from causing
         * another swap in the same time step. This ensures atomic movement.
         */
        ++j;
      }
    }
  }
  
  // Print the final state of the queue.
  for (int i = 0; i < n; ++i) {
    if (v[i])
      printf("B");
    else
      printf("G");
  }
  printf("\n");

  return 0;
}
