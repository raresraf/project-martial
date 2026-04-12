/**
 * @file 116A.CPP
 * @brief Logic for determining the minimum required capacity of a transit vehicle.
 * 
 * Logic: Simulates passenger flow across N stations. Tracks the instantaneous 
 * occupancy after each station exchange (exits and entries) and identifies the 
 * global maximum occupancy encountered.
 * 
 * Algorithm: One-pass state tracking followed by set-wide maximization via sorting.
 * Time Complexity: $O(N \log N)$ where N is the number of stations (due to sorting).
 * Space Complexity: $O(N)$ to store the occupancy state at each station.
 */

#include <bits/stdc++.h>

using namespace std;

int main() {
  int n, current_occupancy = 0;
  // Functional Utility: Reads total transit nodes (stations).
  if (scanf("%d", &n) != 1) return 0;
  
  int a, b;
  // Data Structure: Vector to store the occupancy state history.
  vector<int> states;

  /**
   * Block Logic: Main flow simulation loop.
   * Invariant: Updates current_occupancy based on net passenger flux at each station.
   */
  for (int i = 0; i < n; i++) {
    // a: Passengers exiting, b: Passengers entering.
    scanf("%d%d", &a, &b);
    
    /**
     * Functional Utility: State update.
     * Logic: Subtracts egress before adding ingress to reflect the headcount 
     * after the exchange completes.
     */
    current_occupancy = current_occupancy - a + b;
    
    // Archiving: Records the instantaneous state for post-simulation analysis.
    states.push_back(current_occupancy);
  }

  /**
   * Functional Utility: Global peak detection.
   * Logic: Orders states to identify the high-water mark (minimum sufficient capacity).
   */
  sort(states.begin(), states.end());
  reverse(states.begin(), states.end());
  
  // Output the peak occupancy observed during the simulation.
  cout << states[0] << "\n";

  return 0;
}
/**
 * Note: The $O(N \log N)$ approach is less efficient than $O(N)$ inline maximization 
 * but was documented as implemented.
 */
