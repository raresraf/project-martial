/**
 * @file 158A.CPP
 * @brief Counts participants advancing to the next round of a competition.
 *
 * @details
 *   This program solves the "Next Round" problem from Codeforces.
 *   The rule for advancing is that a participant's score must be positive and
 *   greater than or equal to the score of the k-th place participant.
 *   This file uses a comprehensive competitive programming template with many
 *   macros and utilities, most of which are not used for this specific problem.
 *
 * @problem_link https://codeforces.com/contest/158/problem/A
 * @oj Codeforces
 * @verdict AC
 * @date 2020-05-14
 *
 * @author Saikat Sharma
 * @university CSE, MBSTU
 *
 * @algorithm Array Traversal / Simulation
 * @complexity
 *   - Time: O(N), where N is the number of participants, for reading the scores
 *     and iterating through them once.
 *   - Space: O(N) to store the scores in a vector.
 */

// Includes and utilities from a standard competitive programming template.
#include <algorithm>
#include <bitset>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace __gnu_pbds;
typedef long long ll;
typedef unsigned long long ull;

// Macro for I/O optimization.
#define __FastIO                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0)

// A large collection of macros for various purposes, common in competitive programming.
#define __FileRead freopen("input.txt", "r", stdin)
#define __FileWrite freopen("output.txt", "w", stdout)
#define SET(a, v) memset(a, v, sizeof(a))
#define SZ(v) (int)v.size()
#define pii pair<int, int>
#define pil pair<int, ll>
#define pli pair<ll, int>
#define pll pair<ll, ll>
#define debug cout << "######\n"
#define debug1(x) cout << "### " << x << " ###\n"
#define debug2(x, y) cout << "# " << x << " : " << y << " #\n"
#define nl cout << "\n";
#define sp cout << " ";
#define sl(n) scanf("%lld", &n)
#define sf(n) scanf("%lf", &n)
#define si(n) scanf("%d", &n)
#define ss(n) scanf("%s", n)
#define pf(n) scanf("%d", n)
#define pfl(n) scanf("%lld", n)
#define all(v) v.begin(), v.end()
#define rall(v) v.begin(), v.end()
#define srt(v) sort(v.begin(), v.end())
#define r_srt(v) sort(v.rbegin(), v.rend())
#define rev(v) reverse(v.rbegin(), v.rend())
#define Sqr(x) ((x) * (x))
#define Mod(x, m) ((((x) % (m)) + (m)) % (m))
#define max3(a, b, c) max(a, max(b, c))
#define min3(a, b, c) min(a, min(b, c))
#define pb push_back
#define mk make_pair
#define F first
#define S second
#define MAX 100005
#define INF 1000000009
#define MOD 1000000007

// Template aliases for advanced data structures.
template <class T>
using min_heap = priority_queue<T, std::vector<T>, std::greater<T>>;
template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

// Utility functions for type conversion.
template <typename T> string toString(T Number) {
  stringstream ss;
  ss << Number;
  return ss.str();
}
template <typename T> int toInt(T str) {
  stringstream ss;
  ss << str;
  int num;
  ss >> num;
  return num;
}
ll lcm(ll a, ll b) { return (a / __gcd(a, b)) * b; }
/************************************ Code Start Here ******************************************************/
int main() {
  // Apply I/O optimizations defined in the macro.
  __FastIO;
  //~ cout << setprecision (10) << fixed;

  // n: number of participants, k: the rank to use as a threshold.
  int n, k;
  cin >> n >> k;
  
  // Vector to store the scores of all participants.
  vector<int> vec(n);
  // Invariant: Read N scores into the vector. Input is guaranteed non-increasing.
  for (int i = 0; i < n; i++) {
    cin >> vec[i];
  }

  // Counter for participants who advance.
  int cnt = 0;

  /**
   * @brief Block Logic: Iterate through all scores to count advancing participants.
   */
  for (auto it : vec) {
    /**
     * @brief Pre-condition: Checks if a participant advances.
     * The score must be positive and at least as high as the k-th participant's score.
     * `vec[k-1]` gives the score of the k-th participant (due to 0-based indexing).
     */
    if (it >= vec[k - 1] && it > 0)
      cnt++;
  }
  
  // Output the final count.
  cout << cnt << "\n";
  return 0;
}