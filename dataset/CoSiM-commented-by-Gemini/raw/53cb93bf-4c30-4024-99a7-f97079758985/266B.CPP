//  [amitdu6ey]
#include <bits/stdc++.h>
// A collection of common C++ macros for competitive programming.
#define hell 1000000009
#define bug(x) cout << "$ " << x << " $" << endl
#define ll long long
#define pb push_back
#define mp make_pair
#define tr(z) for (auto it = z.begin(); it != z.end(); it++)
#define loop(i, a, b) for (long long i = a; i < b; i++)
#define vbool vector<bool>
#define vvbool vector<vector<bool>>
#define vchar vector<char>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vll vector<long long>
#define vvl vector<vector<long long>>
#define ninf numeric_limits<int>::min()
#define pinf numeric_limits<int>::max()
using namespace std;

/**
 * @brief Solves the "Queue at the School" problem (Codeforces 266B).
 * @details This function simulates the state of a queue of boys ('B') and
 * girls ('G') over 't' seconds. In each second, every boy standing immediately
 * in front of a girl swaps places with her.
 *
 * @algorithm
 * The simulation is performed in discrete time steps.
 * 1. The initial queue configuration is read.
 * 2. An outer loop runs `t` times, representing each second.
 * 3. An inner loop performs a single pass over the queue for one time step.
 *    - If a 'B' is found at index `i` and a 'G' at `i+1`, they are swapped.
 *    - Crucially, after a swap, the loop counter `i` is incremented. This
 *      ensures that a single boy does not move more than one position per
 *      time step, correctly modeling the simultaneous nature of the swaps.
 * 4. After the simulation, the final queue configuration is printed.
 *
 * @complexity Time: O(n*t), Space: O(n).
 */
void solve() {
  ll n, t;
  string str;
  cin >> n >> t;
  cin >> str;

  // The string is copied into a vector of characters. This is an extra step;
  // the operations could be performed directly on the string.
  vchar s(n);
  loop(i, 0, n) { s[i] = str[i]; }

  /**
   * @block
   * @brief Outer loop simulating the passage of 't' seconds.
   */
  while (t--) {
    /**
     * @block
     * @brief Inner loop performing all swaps for a single second.
     */
    loop(i, 0, n - 1) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        // A boy is in front of a girl, so they swap.
        s[i] = 'G';
        s[i + 1] = 'B';
        // Increment 'i' to prevent the same boy from swapping again in this time step.
        i++;
      }
    }
  }

  // Print the final state of the queue.
  loop(i, 0, n) { cout << s[i]; }
  return;
}

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  cout.tie(0);
  cin.tie(0);
  
  // Standard competitive programming template for handling multiple test cases,
  // though it's fixed to one case here.
  int test_cases = 1;
  // cin>>test_cases;
  while (test_cases--) {
    solve();
  }
  return 0;
}
