/**
 * @file 116A.CPP
 * @brief Analysis of passenger flux across multiple transit nodes.
 * 
 * Logic: Reads station-wise entry and exit data into a static buffer. Performs 
 * a look-ahead accumulation of passenger net change across adjacent stations.
 * Note: The final output is the total station count (N) rather than the peak occupancy.
 * 
 * Time Complexity: $O(N)$ where N is the number of stations.
 * Space Complexity: $O(1)$ auxiliary space beyond the fixed-size event buffer.
 */

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace std;

int main() {
  int n, m = 0, z = 0;
  // Functional Utility: Reads the total number of transit nodes (n).
  cin >> n;
  
  // Data Structure: Pre-allocated buffer for station events (exits, entries).
  int a[1000][2] = {0};
  for (int i = 0; i < n; i++)
    cin >> a[i][0] >> a[i][1];

  /**
   * Block Logic: Accumulates passenger flow with a look-ahead dependency.
   * Logic: Calculates a running delta (z) based on current entries and next station exits.
   * Warning: Accesses a[i+1] which requires careful bounds management for i = n-1.
   */
  for (int i = 0; i < n; i++) {
    // Functional Utility: Computes net flow between current entry and next exit.
    z += a[i][1] - a[i + 1][0];
    // State Modification: Updates subsequent station data with accumulated delta.
    a[i + 1][1] += z;
    // Logic: Identifies local maximums in the processed flow.
    m = max(z, a[i][1]);
  }
  
  /**
   * Functional Utility: Terminates by reporting the cardinality of the station set.
   */
  cout << n;
  
  return 0;
}
