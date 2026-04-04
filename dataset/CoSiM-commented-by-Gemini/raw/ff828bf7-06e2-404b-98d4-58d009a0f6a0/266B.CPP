/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem (266B) on Codeforces.
 * @author @tridib_2003
 *
 * @details This program simulates the changes in a queue of boys ('B') and
 * girls ('G') over 't' seconds. In each second, any boy standing immediately
 * in front of a girl swaps places with her. This implementation performs
 * the swaps in-place.
 *
 * Link: https://codeforces.com/contest/266/problem/B
 *
 * Algorithm:
 * 1. Read the queue length 'n', simulation time 't', and the initial queue 'q'.
 * 2. Loop 't' times to simulate each second.
 * 3. In each time step, iterate through the queue string from left to right.
 *    a. If a 'B' is found at position 'i' and a 'G' at 'i+1', swap them.
 *    b. After a swap, increment the loop counter 'i' an extra time. This is
 *       to ensure that the newly moved 'B' at position 'i+1' doesn't get
 *       to move again in the same time step, correctly modeling one move per second.
 * 4. After 't' seconds, print the final state of the queue.
 *
 * Time Complexity: O(T * N), where T is the time and N is the queue length.
 * Space Complexity: O(N) to store the queue string. The swaps are in-place,
 * so no extra space proportional to N is used during the simulation.
 */
#include <bits/stdc++.h>
using namespace std;

// Common macros used in competitive programming.
#define ll long long
#define PI 3.1415926535897932384626
#define MOD 1000000007
#define vi vector<int>
#define mk(arr, n, type) type *arr = new type[n];
#define FOR(a, c) for (int(a) = 0; (a) < (c); (a)++)
#define w(x)                                                                   \
  int x;                                                                       \
  cin >> x;                                                                    \
  while (x--)
#define FIO                                                                    \
  ios::sync_with_stdio(false);                                                 \
  cin.tie(0);                                                                  \
  cout.tie(0);

int main() {

  FIO;

  int tc = 1;
  // The code is structured for multiple test cases, but only runs once.
  // cin >> tc;
  while (tc--) {

    int n, t;
    cin >> n >> t;

    string q;
    cin >> q;

    /**
     * @brief Outer loop to simulate the 't' seconds of time.
     */
    while (t--) {

      /**
       * @brief Inner loop to perform one pass of swaps for a single time step.
       */
      for (int i = 0; i < q.length() - 1; ++i) {
        if (q[i] == 'B' && q[i + 1] == 'G') {
          // Perform an in-place swap of the adjacent 'B' and 'G'.
          q[i] = 'G';
          q[i + 1] = 'B';
          // Skip the next position to prevent the swapped 'B' from moving again
          // in the same time step.
          ++i;
        }
      }
    }

    cout << q << '\n';
  }

  return 0;
}