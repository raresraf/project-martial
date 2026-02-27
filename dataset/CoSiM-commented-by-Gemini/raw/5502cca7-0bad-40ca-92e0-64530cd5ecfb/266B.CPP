/**
 * @file 266B.CPP
 * @brief Solves the "Queue at the School" problem from Codeforces.
 * @details This program simulates a queue of boys ('B') and girls ('G').
 * In each second, any boy standing immediately in front of a girl will swap
 * places. The program determines the final state of the queue after a
 * given amount of time.
 *
 * @algorithm
 * A discrete time-step simulation is used.
 * 1. Read the queue length `n`, time `t`, and the initial queue state `s`.
 * 2. An outer loop runs `t` times to simulate each second passing.
 * 3. An inner loop performs a single pass over the queue. For each position `i`:
 *    a. If a 'B' is at `s[i]` and a 'G' is at `s[i+1]`, they swap.
 *    b. After a swap, the loop counter `i` is manually incremented. This is a
 *       critical step to ensure that the newly moved boy does not swap again
 *       in the same time step, correctly modeling simultaneous swaps.
 * 4. After the simulation, the final queue configuration is printed.
 *
 * @complexity
 * Time Complexity: O(t * n), due to the nested loops for simulation.
 * Space Complexity: O(n) to store the queue string.
 */
#include <bits/stdc++.h>
using namespace std;

// Common competitive programming macros.
#define faster ios_base::sync_with_stdio(false), cin.tie(NULL), cout.tie(NULL)
#define mp make_pair
#define mod 1000000007
#define qmod 998244353
#define endl "\n"
#define pb push_back
#define ff first
#define ss second

typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

// Note: These constants are defined but not used in this specific problem.
const int MOD = 1e9 + 7;
const int INF = 1e9 + 5;
const ll LINF = LLONG_MAX;

int main() {
  int n; // The number of children in the queue.
  cin >> n;
  int t; // The time in seconds to simulate.
  cin >> t;
  string s; // The string representing the queue.
  cin >> s;

  /**
   * @block
   * @brief Outer loop to simulate 't' seconds of time.
   */
  while (t--) {
    /**
     * @block
     * @brief Inner loop to perform all swaps for a single time step.
     */
    for (int i = 0; i < s.size() - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // Swap the adjacent boy and girl.
        s[i] = 'G';
        s[i + 1] = 'B';
        // Increment 'i' to prevent the boy who just moved from swapping again
        // in the same time step. This is key to the simulation's logic.
        i++;
      }
    }
  }
  
  // Print the final arrangement of the queue.
  cout << s;
}
