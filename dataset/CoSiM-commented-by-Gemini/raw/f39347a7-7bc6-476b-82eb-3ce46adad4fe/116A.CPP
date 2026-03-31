/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem by tracking maximum passenger count.
 *
 * This program calculates the minimum capacity a tram needs to have. It simulates
 * passenger flow over `n` stops. At each stop, `a` passengers exit and `b`
 * passengers enter. The program keeps a running total of the current number of
 * passengers and tracks the maximum number reached at any point. This maximum
 * value is the required minimum capacity.
 *
 * @author f39347a7-7bc6-476b-82eb-3ce46adad4fe
 *
 * @algorithm Simulation and Maximum Value Tracking
 * @complexity
 * Time: O(N), where N is the number of stops, due to the single loop.
 * Space: O(1), for storing a few variables.
 */
#include <bits/stdc++.h>
// Define common macros for competitive programming
#define ll long long int
#define loop(k) for (i = 0; i < k; ++i)
#define loop2(k, l) for (j = k; j < l; ++j)
#define mod 1000000007
using namespace std;
int main() {
  // Fast I/O optimization for C++ streams.
  std::ios_base::sync_with_stdio(false); // cin.tie(NULL);
  
  // `n`: number of stops. `i`: loop counter (used by macro).
  int n, i;
  cin >> n;
  
  // `a`, `b`: exiting and entering passengers.
  // `count`: current passengers on the tram.
  // `d`: maximum passengers seen so far (the required capacity).
  int a, b, count = 0, d = 0;
  
  /**
   * @brief Loop to process each of the `n` stops.
   * The `loop(n)` macro expands to a standard for-loop.
   */
  loop(n) {
    cin >> a >> b;
    // Update the current number of passengers.
    count += (b - a);
    // Update the maximum capacity if the current count is higher.
    d = max(d, count);
  }
  
  // Print the final result, which is the maximum capacity needed.
  cout << d;
  cout << "
";
  return 0;
}