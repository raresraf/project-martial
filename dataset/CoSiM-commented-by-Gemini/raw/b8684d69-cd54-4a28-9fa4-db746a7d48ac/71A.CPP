/**
 * @file 71A.CPP
 * @brief Solution to a greedy resource allocation problem.
 *
 * @warning The filename `71A.CPP` is misleading and does not correspond to the
 * "Way Too Long Words" problem. The code solves a different, unknown problem.
 *
 * @details
 * This program solves a problem where a limited resource `a` must be used to
 * maximize profit from two types of actions.
 * - Action X: Can be performed up to `b` times, yields `x` profit per action.
 * - Action Y: Can be performed up to `c` times, yields `y` profit per action.
 * Each action (of either type) appears to consume 2 units of resource `a`.
 *
 * The code implements a greedy strategy. It first compares the profits `x` and `y`.
 * It then allocates the resource `a` to as many instances of the more profitable
 * action as possible. If any resource `a` remains, it is then used for the less
 * profitable action.
 *
 * The implementation is inefficient, using `while` loops to decrement counts instead
 * of performing a direct O(1) calculation with division.
 *
 * Algorithm: Greedy resource allocation.
 * Time Complexity: O(B+C) in the worst case, due to the inefficient `while` loops.
 *                  An optimal implementation would be O(1).
 * Space Complexity: O(1).
 */
#include <bits/stdc++.h>
#define ll long long int
using namespace std;
// A large number of global variables are declared, a common but inadvisable
// practice in competitive programming. Most are unused in this specific problem.
ll ar[311111], br[211111];
vector<ll> v;
map<ll, ll> M;
string s;
ll cnt, Min = 999999999, Max = 0, ans = 0, a, b, c, d, e, k, n, m, x, y, z;

int main() {
  ll test;
  cin >> test;
  while (test--) {
    ans = 0;
    // a: available resource
    // b: count of action X
    // c: count of action Y
    // x: profit of action X
    // y: profit of action Y
    cin >> a >> b >> c;
    cin >> x >> y;

    // Greedy choice: Prioritize the action with the higher profit.
    if (x >= y) {
      // --- Case 1: Action X is more profitable ---

      // Inefficiently calculate how many of action X can be performed.
      // A direct calculation like `ll num_x = min(b, a / 2);` would be O(1).
      if (a > b * 2) {
        // This branch seems to assume there's enough resource `a` for all `b` actions.
        ans += b * x;
        a -= b * 2;
        // After taking all `b` actions, try to use remaining `a` for action Y.
        if (a > c * 2) {
          ans += c * y;
        } else {
          // This while loop is an inefficient way to do `ans += min(c, a/2) * y;`
          while (c > 0) {
            if (c * 2 <= a) {
              ans += c * y;
              break;
            }
            c--;
          }
        }
      } else {
        // This while loop is an inefficient way to do `ans += (a/2) * x;`
        while (b > 0) {
          if (b * 2 <= a) {
            ans += b * x;
            break;
          }
          b--;
        }
      }
    } else {
      // --- Case 2: Action Y is more profitable (symmetric logic) ---
      if (a > c * 2) {
        ans += c * y;
        a -= c * 2;
        if (a > b * 2) {
          ans += b * x;
        } else {
          // Inefficient loop.
          while (b > 0) {
            if (b * 2 <= a) {
              ans += b * x;
              break;
            }
            b--;
          }
        }
      } else {
        // Inefficient loop.
        while (c > 0) {
          if (c * 2 <= a) {
            ans += c * y;
            break;
          }
          c--;
        }
      }
    }
    cout << ans << endl;
  }

  return 0;
}