/**
 * @file 1221C.CPP
 * @brief Solves a team formation optimization problem using binary search.
 *
 * This program determines the maximum number of teams that can be formed given
 * a number of coders (c), mathematicians (m), and other experts (x). Each team
 * must have one coder, one mathematician, and one other expert. The pool of
 * "other experts" can be formed from the remaining coders, mathematicians,
 * or the initial 'x' experts.
 *
 * @algorithm Binary Search on the Answer. The feasibility of forming 'k' teams
 *            is a monotonic function, making it suitable for binary search.
 * @time_complexity O(log(min(c, m))), as the binary search is performed on the
 *                  number of possible teams, which is bounded by the minimum
 *                  of coders and mathematicians.
 * @space_complexity O(1).
 */
//  [amitdu6ey]
// g++ -std=c++11 -o2 -Wall filename.cpp -o filename
#include <bits/stdc++.h>
#define hell 1000000009
#define bug1(x) cout << "$ " << x << " $" << endl
#define bug2(x) cout << "% " << x << " %" << endl
#define bug3(x) cout << "# " << x << " #" << endl
#define ll long long
#define pb push_back
#define mp make_pair
#define tr(z) for (auto it = z.begin(); it != z.end(); it++)
#define rloop(i, a, b) for (long long i = a; i >= b; i--)
#define loop(i, a, b) for (long long i = a; i < b; i++)
#define vbool vector<bool>
#define vvbool vector<vector<bool>>
#define vchar vector<char>
#define vi vector<int>
#define vvi vector<vector<int>>
#define pll pair<long long, long long>
#define vll vector<long long>
#define vvl vector<vector<long long>>
#define ninf numeric_limits<long long>::min()
#define pinf numeric_limits<long long>::max()
#define comp(a, b) (abs(a - b) < 1e-9) // to compare doubles
using namespace std;

/**
 * @brief Checks if it's possible to form a given number of teams.
 * @param c The number of available coders.
 * @param m The number of available mathematicians.
 * @param x The number of available other experts.
 * @param ts The number of teams to check for feasibility.
 * @return True if 'ts' teams can be formed, false otherwise.
 */
bool team_possible(ll c, ll m, ll x, ll ts) {
  // Condition 1: We must have enough coders and mathematicians for 'ts' teams.
  if (ts > min(c, m)) {
    return false;
  }
  // After forming 'ts' teams, each with one coder and one mathematician...
  m -= ts;
  c -= ts;
  // Condition 2: The remaining people must be enough to fill the third spot in each of the 'ts' teams.
  if (ts <= c + m + x) {
    return true;
  }
  return false;
}

/**
 * @brief Finds the maximum number of teams that can be formed for a single test case.
 */
void solve() {
  ll c, m, x;
  cin >> c >> m >> x;
  ll l = 0, r = 1e9, mid = (l + r) / 2;
  ll ans = 0;
  /**
   * @brief Binary search for the maximum possible number of teams.
   * The search space is from 0 to 10^9 teams. For each 'mid' value,
   * it checks if that many teams are possible. If so, it tries for more
   * teams (l = mid + 1); otherwise, it reduces the search space (r = mid - 1).
   */
  while (l <= r) {
    mid = l + (r - l) / 2; // Avoids potential overflow
    if (team_possible(c, m, x, mid)) {
      ans = mid; // 'mid' is a possible answer, store it.
      l = mid + 1; // Try for a larger number of teams.
    } else {
      r = mid - 1; // 'mid' is not possible, reduce the upper bound.
    }
  }
  cout << ans << "\n";
}

int main() {
  // Fast I/O setup
  ios_base::sync_with_stdio(0);
  cout.tie(0);
  cin.tie(0);
  // freopen("input.txt","r",stdin);
  // freopen("output.txt","w",stdout);
  ll test_cases = 1;
  cin >> test_cases;
  while (test_cases--) {
    solve();
  }
  return 0;
}
