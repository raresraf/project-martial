/**
 * @file 266B.CPP
 * @brief An attempted solution for the "Queue at the School" problem (Codeforces 266B).
 * @author competitive-programmer
 *
 * @details This program attempts to simulate the process of rearranging a queue
 * of boys ('B') and girls ('G'). In each time step, any boy standing
 * immediately in front of a girl should swap places with her.
 *
 * @warning This implementation is critically flawed. It results in undefined
 * behavior due to an out-of-bounds array access.
 *
 * Flawed Algorithm:
 * 1. Read `n` (length) and `t` (time steps).
 * 2. Loop `t` times to simulate time.
 * 3. In the inner loop, it iterates from `i = 0` to `n-1`.
 * 4. The condition `if (s[i] == 'G' && s[i - 1] == 'B')` attempts to find
 *    'BG' pairs.
 * 5. **Bug:** When `i` is 0, the expression `s[i-1]` reads from index -1, which
 *    is an out-of-bounds access and causes undefined behavior.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  
  // Outer loop for simulating 't' seconds.
  while (t--) {
    // Inner loop to iterate through the queue.
    for (int i = 0; i < n; i++) {
      /**
       * @bug Critical Error: Out-of-bounds access. When `i` is 0, this
       * condition attempts to access `s[i-1]`, i.e., `s[-1]`, which leads
       * to undefined behavior. The check should be for `s[i] == 'B'` and
       * `s[i+1] == 'G'`.
       */
      if (s[i] == 'G' && s[i - 1] == 'B') {
        swap(s[i], s[i - 1]);
        // The `i++` is intended to skip the next character to ensure atomic
        // swaps, but it doesn't function correctly with the flawed condition.
        i++;
      }
    }
  }
  cout << s << endl;
}
