/**
 * @file 670A.CPP
 * @brief Solution to a competitive programming problem involving frequency and MEX (Minimum Excluded value).
 *
 * @details
 * This program solves a problem for multiple test cases. For each test case, it's
 * given an array of integers. The task is to find the sum of two values:
 * 1. The smallest non-negative integer that does NOT appear in the array at all (MEX).
 * 2. The smallest non-negative integer that does NOT appear in the array at least twice.
 *
 * @algorithm
 * The algorithm for each test case is as follows:
 * 1. Count the frequencies of all numbers in the input array using an unordered_map.
 * 2. Populate two unordered_sets:
 *    - `c`: Contains all unique numbers present in the input array.
 *    - `b`: Contains only the numbers that appear two or more times.
 * 3. Calculate the first value (`c2` in the code): Find the MEX of set `c`. This is done
 *    by checking for 0, 1, 2, ... until a number is not found in the set.
 * 4. Calculate the second value (`c1` in the code): Find the MEX of set `b`. This
 *    finds the smallest non-negative integer that is not duplicated.
 * 5. The final answer is the sum of these two MEX values.
 *
 * @complexity
 * - Time Complexity: O(N) on average per test case, where N is the number of elements.
 *   This is dominated by reading input, building the frequency map, and finding the MEX values.
 *   The total time for T test cases is O(sum of all Ns).
 * - Space Complexity: O(N) per test case, for storing the input array, frequency map, and sets.
 */

#include <bits/stdc++.h>

using namespace std;
using ll = long long;
using vll = vector<ll>;
using vvll = vector<vll>;
using vvvll = vector<vvll>;

#define REP(i, n, m) for (ll i = n; i < (ll)m; ++i)
#define IREP(i, n, m) for (ll i = n - 1; i >= m; --i)
#define rep(i, n) REP(i, 0, n)
#define irep(i, n) IREP(i, n, 0)
#define all(v) v.begin(), v.end()
#define vprint(v)                                                              
  for (auto e : v) {                                                           
    cout << e << " ";                                                          
  };                                                                           
  cout << endl;
#define vvprint(vv)                                                            
  for (auto v : vv) {                                                          
    vprint(v)                                                                  
  };

/**
 * @brief Main function and entry point of the program.
 */
int main() {
  // Fast I/O setup
  cin.tie(0);
  ios::sync_with_stdio(false);
  cout << setprecision(20);

  ll t;
  cin >> t;

  // Main loop to handle all test cases.
  rep(i, t) {
    ll n;
    cin >> n;
    vll a(n);
    rep(j, n) cin >> a[j];

    // Use an unordered_map to count the frequency of each number.
    unordered_map<ll, ll> m;
    rep(j, n)++ m[a[j]];

    // `b` will store numbers that appear at least twice.
    // `c` will store all unique numbers.
    unordered_set<ll> b, c;
    for (auto p : m) {
      if (p.second >= 2) {
        b.insert(p.first);
        c.insert(p.first);
      } else
        c.insert(p.first);
    }

    ll c1 = 0, c2 = 0;
    
    /**
     * @brief Calculate MEX for numbers that appear at least twice.
     * `c1` will be the smallest non-negative integer not present in set `b`.
     */
    while (b.count(c1) > 0)
      ++c1;
    
    /**
     * @brief Calculate MEX for all unique numbers.
     * `c2` will be the smallest non-negative integer not present in set `c`.
     */
    while (c.count(c2) > 0)
      ++c2;
    
    // The answer is the sum of the two calculated MEX values.
    cout << c1 + c2 << endl;
  }
}
