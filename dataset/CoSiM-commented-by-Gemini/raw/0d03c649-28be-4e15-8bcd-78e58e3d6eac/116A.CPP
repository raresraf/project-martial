/**
 * @file 116A.CPP
 * @brief Solves the "Tram" problem using a two-pass approach.
 * @details This program calculates the minimum required capacity of a tram. It first reads all
 * passenger data (exiting and entering counts for each stop) into two separate vectors.
 * After all input is stored, it performs a second pass to simulate the journey, tracking the
 * current number of passengers and updating the maximum occupancy seen so far.
 *
 * @author amitdu6ey
 *
 * @algorithm Input pre-loading followed by simulation.
 * @complexity
 * Time complexity: O(N), where N is the number of stops. The program uses two separate loops, each running N times.
 * Space complexity: O(N) to store the entrance and exit counts in two vectors.
 */

#include <bits/stdc++.h>
#define bug(x) cout << "$ " << x << " $" << endl
#define ll long long
#define pb push_back
#define mp make_pair
#define tr(z) for (auto it = z.begin(); it != t.end(); it++)
#define loop(i, a, b) for (long long i = a; i < b; i++)
#define vbool vector<bool>
#define vvbool vector<vector<bool>>
#define vchar vector<char>
#define vi vector<int>
#define vvi vector<vector<int>>
#define vll vector<long long>
#define vvl vector<vector<long long>>

using namespace std;

/**
 * @brief Contains the core logic for solving the problem.
 */
void solve() {
  int n;
  cin >> n;
  vector<int> a(n); // Vector to store exiting passenger counts.
  vector<int> b(n); // Vector to store entering passenger counts.
  
  /**
   * @brief First pass: Read and store all passenger data.
   */
  for (int i = 0; i < n; i++) {
    cin >> a[i] >> b[i];
  }
  
  // sum: current number of passengers.
  // max: maximum number of passengers observed. Initialized to -1, which is safe.
  int sum = 0, max = -1;
  
  /**
   * @brief Second pass: Simulate the tram's journey and find the peak occupancy.
   */
  for (int i = 0; i < n; i++) {
    sum += b[i] - a[i]; // Update the passenger count after a stop.
    if (sum > max)
      max = sum;
  }
  cout << max;
  return;
}

/**
 * @brief Main function with a typical competitive programming test case loop.
 */
int main() {
  int test_cases = 1;
  // The test case input is commented out, so it runs only once.
  // cin>>test_cases;
  while (test_cases--) {
    solve();
  }
  return 0;
}
