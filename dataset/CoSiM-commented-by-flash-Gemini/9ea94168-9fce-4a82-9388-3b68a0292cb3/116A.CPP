/**
 * @file 116A.CPP
 * @brief Simulation of a public transit system to determine the minimum required capacity.
 * 
 * Algorithm: One-pass simulation of passenger flow (inflow/outflow) at multiple stations.
 * Time Complexity: $O(N)$ where N is the number of stations.
 * Space Complexity: $O(1)$ as it only tracks the current and maximum occupancy.
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#define vc vector<int>
#define gcd(x, y) __gcd(x, y)
#define lcm(a, b) (a * (b / gcd(a, b)))
#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)
#define MAX_VAL 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  int n, a, b;

  /**
   * Functional Utility: Reads the total number of stations in the route.
   */
  while (scanf("%d", &n) == 1) {
    // Current occupancy of the tram.
    int sum = 0;
    // High-water mark for occupancy (minimum required capacity).
    int maxsum = 0;

    /**
     * Block Logic: Simulates the exchange of passengers at each individual station.
     * Invariant: sum >= 0 (Physical constraint: outflow cannot exceed current occupancy).
     */
    for (int i = 0; i < n; i++) {
      // a: passengers exiting, b: passengers entering.
      cin >> a >> b;
      // Functional Utility: Updates the current occupancy by net change.
      sum = (sum - a) + b;

      // Logic: Dynamically updates the peak capacity observed during the simulation.
      if (sum > maxsum)
        maxsum = sum;
    }
    // Output the minimum capacity that was sufficient for all stations.
    printf("%d\n", maxsum);
  }
  return 0;
}
