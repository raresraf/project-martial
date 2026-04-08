/*
  Yadi Qian
*/

/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces (Problem 266B).
 * @details This program simulates the changing positions in a queue of boys ('B')
 * and girls ('G') over `t` seconds. In each second, any boy standing immediately
 * in front of a girl swaps places with her.
 *
 * @algorithm
 * 1. Read the number of children `n`, the time in seconds `t`, and the initial queue
 *    as a string.
 * 2. The outer loop runs `t` times, simulating the passage of each second.
 * 3. The inner loop iterates through the queue to find adjacent pairs of a boy
 *    followed by a girl ('BG').
 * 4. When a 'BG' pair is found, they are swapped to 'GB'.
 * 5. After a swap, the inner loop's counter is incremented by an extra step (`j++`).
 *    This is to ensure that a boy who has just moved does not move again in the
 *    same time step, correctly modeling the problem's constraints.
 * 6. After the outer loop completes, the final state of the queue is printed.
 *
 * @time_complexity O(n * t), where `n` is the queue length and `t` is the time.
 * @space_complexity O(n) for storing the queue string.
 */
#include <bits/stdc++.h>

using namespace std;

const double pi = acos(-1.0);
typedef pair<int, int> PII;
typedef vector<int> VI;
typedef vector<string> VS;
typedef vector<PII> VII;
typedef vector<VI> VVI;
typedef map<int, int> MPII;
typedef set<int> SETI;
typedef multiset<int> MSETI;
typedef long int int32;
typedef unsigned long int uint32;
typedef long long int int64;
typedef unsigned long long int uint64;

int main() {
  // Fast I/O
  ios_base::sync_with_stdio(false);
  cin.tie(NULL);

  int n, t;
  string line;
  cin >> n >> t >> line;

  /**
   * @block_logic Outer loop simulates the progression of time for `t` seconds.
   */
  for (int i = 0; i < t; i++) {
    /**
     * @block_logic Inner loop performs a single pass over the queue, swapping
     * adjacent 'B' and 'G' pairs.
     */
    for (int j = 0; j < n - 1; j++) {
      if (line[j] == 'B' && line[j + 1] == 'G') {
        // Swap the boy and girl.
        line[j] = 'G';
        line[j + 1] = 'B';
        // @inline: Increment `j` to prevent the same boy from moving twice
        // in the same time step.
        j++;
      }
    }
  }

  cout << line;

  return 0;
}