/**
 * @file 266B.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Queue at the School").
 * This program simulates a state-reordering process in a queue of boys ('B')
 * and girls ('G') based on a local adjacency rule over discrete time steps.
 *
 * @algorithm Iterative single-pass swap simulation.
 * @time_complexity O(T * N), where T is the number of time steps and N is the length of the queue.
 * @space_complexity O(N) to store the queue sequence.
 */
#include <bits/stdc++.h>
using namespace std;
#define ll long long
#define ff first
#define ss second
#define pb push_back
#define mod 1000000007
#define pi acos(-1)

int main() {
  // freopen("input.txt", "r", stdin);
  // freopen("output.txt", "w", stdout);

  /**
   * Functional Utility: Decouples C++ streams from standard C streams and flushes
   * to optimize high-throughput I/O required for competitive programming.
   */
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n, t;
  string s;

  cin >> n >> t;

  cin >> s;

  /**
   * @block
   * @brief Orchestrates the temporal progression of the simulation for `t` seconds.
   * Invariant: At the start of each iteration, `s` represents the queue state
   * at the beginning of the current time step.
   */
  while (t--) {
    /**
     * @block
     * @brief Performs a single-pass sweep to resolve inversions ('BG' pairs).
     * Logic: Identifies adjacent 'B' and 'G' pairs where the boy precedes the
     * girl and performs a local state transition (swap).
     */
    for (int i = 0; i < s.length(); i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        swap(s[i], s[i + 1]);
        /**
         * @inline
         * Skips the next index to prevent a single 'G' that has just moved
         * from being involved in another swap in the same time step. This
         * ensures atomic movement (one person moves at most one position
         * per time step).
         */
        i++;
      }
    }
  }

  // Output the final state of the queue after `t` seconds.
  cout << s << "\n";

  return 0;
}