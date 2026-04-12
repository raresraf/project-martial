/**
 * @file 116A.CPP
 * @brief Logic for determining the peak occupancy of a public transit vehicle.
 * 
 * Logic: Simulates the passenger flow (inflow and outflow) across N stations. 
 * The peak occupancy encountered during the simulation represents the minimum 
 * required capacity for the vehicle.
 * 
 * Algorithm: Single-pass state tracking with high-water mark detection.
 * Time Complexity: $O(N)$ where N is the number of stations.
 * Space Complexity: $O(1)$ constant auxiliary space.
 */

#include <bits/stdc++.h>

// Functional Utility: Standardized loop macros for competitive programming.
#define _for(i, a, b) for (int i = (a), _b_ = (b); i < _b_; i++)
#define _fod(i, a, b) for (int i = (a), _b_ = (b); i > _b_; i--)
#define _it(i, v) for (typeof((v).begin()) i = (v).begin(); i != (v).end(); ++i)
#define _all(v) v.begin(), v.end()
#define __(v) memset(v, 0, sizeof(v))

using namespace std;
typedef long long LL;
typedef unsigned long long ULL;

// Functional Utility: Operator extension for vector population.
template <typename T> vector<T> &operator+=(vector<T> &v, T x) {
  v.push_back(x);
  return v;
}

/**
 * @brief core simulation logic for passenger capacity evaluation.
 */
void solve() {
  int n, max_capacity = 0, current_occupancy, exiting, entering;
  // Reads total transit nodes.
  if (!(cin >> n)) return;

  /**
   * Block Logic: Main flow simulation loop.
   * Invariant: current_occupancy represents the headcount after processing station i.
   */
  _for(i, 0, n) {
    // exiting: passengers leaving, entering: passengers joining.
    cin >> exiting >> entering;
    
    // Logic: initialization for the first station.
    if (!i) {
      current_occupancy = entering;
      max_capacity = entering;
    } else {
      /**
       * Functional Utility: State update based on flux.
       * Logic: Subtracts exiting count before adding entering count to reflect 
       * instantaneous occupancy correctly.
       */
      current_occupancy -= exiting;
      current_occupancy += entering;
      
      // High-water mark logic: identifies the required capacity bottleneck.
      if (current_occupancy > max_capacity)
        max_capacity = current_occupancy;
    }
  }
  
  // Output the minimum capacity sufficient for the entire route.
  cout << max_capacity << endl;
}

int main() {
  /**
   * Functional Utility: Optimizes standard I/O throughput.
   */
  ios_base::sync_with_stdio(0);
  cin.tie(0);

#ifdef HIEUNV
  freopen("input.txt", "r", stdin);
  freopen("output.txt", "w", stdout);
#endif

  solve();
  
  return 0;
}
