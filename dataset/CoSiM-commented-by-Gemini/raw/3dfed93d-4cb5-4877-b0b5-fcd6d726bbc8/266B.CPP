/**
 * @file 266B.CPP
 * @brief Solution for the "Queue at the School" problem (Codeforces 266B).
 * @author TENSHI29
 *
 * @algorithm
 * The problem simulates a queue of boys ('B') and girls ('G'). In each time step,
 * any boy standing directly in front of a girl swaps places with her. This
 * process is repeated for a given number of seconds.
 *
 * The algorithm simulates this process step by step:
 * 1. An outer loop runs `t` times, once for each second.
 * 2. An inner loop performs a single pass over the queue for one time step.
 * 3. Inside the pass, it looks for adjacent "BG" pairs.
 * 4. When a "BG" pair is found at indices `j` and `j+1`, they are swapped to "GB".
 * 5. Crucially, after a swap, the inner loop's index `j` is incremented an
 *    extra time. This ensures that the boy who just moved to `j+1` is not
 *    considered for another swap in the same time step, correctly modeling the
 *    simultaneous nature of the swaps.
 *
 * @complexity
 * - Time Complexity: O(T * N), where T is the number of seconds and N is the
 *   length of the queue.
 * - Space Complexity: O(N) to store the queue string.
 *
 * @warning
 * The code contains a potential out-of-bounds access bug. The inner loop
 * condition `j < n` allows `j` to become `n-1`, at which point `q[j+1]` (i.e., `q[n]`)
 * is accessed, which is past the end of the string. The condition should be
 * `j < n - 1`.
 */
/*=====TENSHI29=====*/
#include <iostream>
//#include<algorithm>
#include <string>
using namespace std;
int main() {
  int n, t; // n: queue length, t: time in seconds.
  // char q[n];
  string q;
  cin >> n >> t;
  // for(int i=0;i<n;i++)
  cin >> q;
  /**
   * @brief Outer loop to simulate `t` seconds of time passing.
   */
  while (t--) {
    /**
     * @brief Inner loop to perform one pass of swaps for a single second.
     * WARNING: Loop condition should be `j < n - 1` to avoid out-of-bounds access.
     */
    for (int j = 0; j < n; j++) {
      // Find a boy standing immediately in front of a girl.
      if (q[j] == 'B' && q[j + 1] == 'G') {
        // swap(q[j],q[j+1]);
        // Swap their positions.
        q[j + 1] = 'B';
        q[j] = 'G';
        // Increment `j` an extra time to prevent the newly moved boy at `j+1`
        // from swapping again in the same time step. This correctly simulates
        // the simultaneous swaps.
        j++;
      }
    }
  }
  // Print the final state of the queue.
  // Note: A simpler `cout << q;` would also work.
  for (int i = 0; i < n; i++)
    cout << q[i];
  return 0;
}
