/**
 * @file 266B.CPP
 * @brief Simulates the swapping of boys and girls in a queue.
 *
 * This program simulates a queue of boys ('B') and girls ('G'). Over a given
 * number of time steps `t`, any boy standing immediately in front of a girl
 * will swap places with her.
 *
 * @see https://codeforces.com/problemset/problem/266/B
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  string str;
  cin >> str;
  /**
   * @brief Time simulation loop.
   *
   * This loop iterates `t` times, representing `t` seconds of time. In each
   * iteration, a pass is made through the queue to perform swaps.
   */
  while (t--) {
    /**
     * @brief Queue pass for swapping.
     *
     * This loop iterates through the queue. If it finds a boy ('B') followed
     * by a girl ('G'), it swaps their positions. The loop counter is then
     * incremented to ensure that the newly swapped girl doesn't immediately
     * swap again in the same time step.
     */
    for (int i = 1; i < n; ++i) {
      if (str[i] == 'G' && str[i - 1] == 'B') {
        str[i] = 'B';
        str[i - 1] = 'G';
        ++i;
      }
    }
  }
  cout << str;
}
