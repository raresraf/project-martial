/**
 * @file 1221C.CPP
 * @brief Solution for the "Perfect Team" problem.
 *
 * @problem We are given 'c' coders, 'm' mathematicians, and 'x' universals.
 * A team consists of 3 people: one coder, one mathematician, and one other person
 * (who can be a coder, mathematician, or universal). The goal is to find the
 * maximum number of teams that can be formed.
 *
 * @algorithm Let 'k' be the number of teams. To form 'k' teams, we must satisfy
 * two main constraints:
 * 1. Specialist Constraint: We need at least 'k' people to fill the coder roles
 *    and 'k' people to fill the mathematician roles. Since we have 'c' coders and
 *    'm' mathematicians, 'k' cannot exceed either of these numbers.
 *    Thus, `k <= min(c, m)`.
 *
 * 2. Total People Constraint: Each of the 'k' teams requires 3 distinct people.
 *    The total number of people available is `c + m + x`. Therefore, the total
 *    number of people used (`3 * k`) cannot exceed the total available.
 *    Thus, `3 * k <= c + m + x`, which simplifies to `k <= (c + m + x) / 3`.
 *
 * To maximize 'k', we must satisfy both constraints. The solution is the
 * minimum of these two upper bounds: `min(min(c, m), (c + m + x) / 3)`.
 *
 * @time_complexity O(1) for each test case.
 * @space_complexity O(1).
 */
#include <bits/stdc++.h>
// Boilerplate macros for competitive programming
#define ll long long int
#define pb push_back
#define F first
#define S second
#define mp make_pair
#define MOD 1000000007
#define vi vector<int>
#define vll vector<ll>
#define pll pair<ll, ll>
#define pii pair<int, int>
#define all(p) p.begin(), p.end()
#define mid(s, e) (s + (e - s) / 2)
#define eb emplace_back
#define ull unsigned long long
// Macro to disable synchronization with C stdio for faster I/O.
#define KAMEHAMEHA ios_base::sync_with_stdio(0);
// Macro for even faster I/O by un-tying cin and cout.
#define RASENGAN                                                               \
  ios_base::sync_with_stdio(0);                                                \
  cin.tie(0);                                                                  \
  cout.tie(0);
using namespace std;
int main() {
  KAMEHAMEHA
  //    #ifdef _soumik
  //        freopen("input.txt", "r", stdin);
  //    #endif
  int q, c, m, x;
  cin >> q;
  // Process all test cases.
  while (q--) {
    cin >> c >> m >> x;
    // 'mn' represents the specialist constraint: we can't have more teams
    // than the number of available coders or mathematicians.
    int mn = min(c, m);
    
    // The final answer is the minimum of two constraints:
    // 1. The total number of people divided by 3 (`(c + m + x) / 3`).
    // 2. The maximum number of teams limited by the scarcest specialist (`mn`).
    cout << min((c + m + x) / 3, mn) << endl;
  }
  return 0;
}
