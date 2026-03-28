/**
 * @file 266B.CPP
 * @brief Simulates a queue rearrangement process.
 *
 * This program simulates a queue of boys ('B') and girls ('G'). Over a given
 * number of time steps (`t`), any boy standing immediately in front of a girl
 * will swap places with her. This process is repeated for the specified duration.
 *
 * @b Algorithm: The simulation uses a nested loop structure. The outer loop
 *    progresses time, and the inner loop sweeps through the queue to perform
 *    swaps. A single pass is made for each time step. When a 'B' followed by
 *    a 'G' is found, they are swapped, and the loop index is incremented to
 *    prevent the newly moved 'G' from swapping again in the same time step.
 *
 * @b Time_Complexity: O(t * n), where 't' is the number of time steps and 'n'
 *    is the length of the queue.
 * @b Space_Complexity: O(n) to store the queue as a string.
 */
#include <bits/stdc++.h>

#define ll long long int
#define MAX (ll)(10e6 + 5)

using namespace std;

int main() {
  /**
   * @brief Standard competitive programming optimization for fast I/O.
   * It unties cin/cout from stdio and disables synchronization.
   */
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  cout.tie(nullptr);
  // freopen("266B.in","r",stdin);
  int n, t;
  string s;
  cin >> n >> t;
  cin >> s;
  
  /**
   * @brief Outer loop for time steps.
   */
  while (t--) {
    /**
     * @brief Inner loop to scan the queue for 'BG' pairs and swap them.
     */
    for (int i = 0; i < n; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        // Increment 'i' to prevent the same 'G' from moving multiple times
        // in one time step.
        i++;
      }
    }
  }
  cout << s << endl;
  return 0;
}