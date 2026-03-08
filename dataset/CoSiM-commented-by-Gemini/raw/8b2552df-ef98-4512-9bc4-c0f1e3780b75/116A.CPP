/**
 * @file 116A.CPP
 * @brief An optimal solution for the "Tram" capacity problem.
 *
 * @details
 * This program calculates the minimum required capacity of a tram by simulating
 * passenger flow. It uses a single-pass, constant-space approach. For each stop,
 * it reads the number of exiting and entering passengers, updates the current
 * passenger count, and immediately compares this count to the maximum found so
 * far. This avoids the need for storing passenger counts for all stops.
 * The code uses a common competitive programming template with macros and typedefs,
 * though most are unused in this specific solution.
 *
 * @section
 * Algorithm: Single-pass simulation with real-time maximum tracking.
 * Time Complexity: O(N), where N is the number of stops.
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>

// Common macro for a for-loop, unused in this solution.
#define FOR(i, a, n) for (int i = (int)(a); i < (int)(n); ++i)
// Common macro for a zero-based for-loop, unused.
#define REP(i, n) FOR(i, 0, n)
// Common macro to get iterators to the beginning and end of a container, unused.
#define all(a) a.begin(), a.end()
// Common macro for push_back, unused.
#define pb push_back

// Typedefs for common long number types, unused in this solution.
typedef unsigned long long llu;
typedef long long ll;
typedef long double ld;

using namespace std;

int main() {
  // curr: tracks the current number of passengers on the tram.
  // n: the total number of stops.
  // m: tracks the maximum number of passengers seen so far (the required capacity).
  int curr = 0, n, m = 0;
  scanf("%d", &n);

  /**
   * @brief A single loop to simulate all stops and find the peak passenger count.
   *
   * Logic: This loop processes one stop at a time. It updates the current
   * passenger count and immediately checks if this new count is a peak.
   * This is an O(1) space approach as it does not store data for each stop.
   */
  while (n--) {
    int a, b; // a: exiting passengers, b: entering passengers.
    scanf("%d %d", &a, &b);
    // Update the current passenger count.
    curr -= a;
    curr += b;
    // Update the maximum capacity if the current count is higher.
    m = max(curr, m);
  }
  // 'm' now holds the highest passenger count recorded, which is the answer.
  cout << m << endl;
  return 0;
}
