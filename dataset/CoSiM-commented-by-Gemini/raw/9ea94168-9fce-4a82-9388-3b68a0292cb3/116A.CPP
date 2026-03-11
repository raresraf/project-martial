/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem.
 *
 * This program calculates the minimum possible capacity of a tram based on the number
 * of passengers exiting and entering at each stop. The goal is to find the maximum
 * number of passengers in the tram at any point in its route.
 *
 * Algorithm: Simulation. It iterates through each stop, updating the current
 *            number of passengers and tracking the maximum count observed.
 *
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1), as it only uses a few variables to store the current state.
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
#define MAX 1000001
#define inf 999999999
#define PI acos(-1)
#define MOD 1000000007

using namespace std;

int main() {
  int n; // Number of tram stops.
  int a; // Number of passengers exiting the tram.
  int b; // Number of passengers entering the tram.

  // The while loop with scanf is a common competitive programming pattern for handling multiple test cases,
  // though for this specific problem, it processes a single test case.
  while (scanf("%d", &n) == 1) {
    int sum = 0; // Current number of passengers in the tram.
    int maxsum = 0; // Tracks the maximum number of passengers at any time.
    
    /**
     * @brief Loop through each stop to simulate passenger flow.
     *
     * Pre-condition: 'n' is the number of stops.
     * Invariant: 'sum' holds the number of passengers after the ith stop.
     */
    for (int i = 0; i < n; i++) {
      cin >> a >> b;
      
      // Update the number of passengers: subtract exiting, add entering.
      sum = (sum - a) + b;

      /**
       * @brief Update the maximum capacity required.
       * Logic: If the current number of passengers exceeds the previously
       *        recorded maximum, update the maximum.
       */
      if (sum > maxsum)
        maxsum = sum;
    }
    // Output the minimum required capacity.
    printf("%d
", maxsum);
  }
  return 0;
}
