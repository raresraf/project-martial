/**
 * @file 1426C.cpp
 * @brief Solution for Codeforces Problem 1426C, "Increase and Copy".
 * @details The problem asks for the minimum number of moves to obtain at least 'n' elements,
 * starting with one element. The allowed moves are: (1) increment the current number of elements by 1,
 * and (2) copy all current elements, effectively doubling the count if you have 'x' elements and you copy them, you get '2x' elements.
 *
 * Algorithm:
 * The solution iterates through possible strategies. A strategy consists of some number of "increase" moves followed by some number of "copy" moves.
 * Let's say we perform `x-1` "increase" moves. This results in `x` elements. The number of moves so far is `x-1`.
 * Now, we have `x` elements and we want to reach at least `n` elements by copying.
 * Each copy operation adds the current number of elements. To reach `n` from `x`, we need to add `n-x` more elements.
 * The number of copy operations required is `ceil((n-x)/x)`.
 * In integer arithmetic, `ceil(a/b)` can be calculated as `(a + b - 1) / b`.
 * So, the number of copies is `((n - x) + x - 1) / x`, which is equivalent to `(n-1)/x`.
 * The total number of moves for a given `x` is `(x - 1) + ((n - x) + x - 1) / x`.
 * The code iterates through `x` from 1 up to `sqrt(n)`. This is an optimization based on the fact that the minimum value of the function `f(x) = x + n/x` is at `x = sqrt(n)`. The expression for the number of moves is similar to this function.
 *
 * Time Complexity: O(sqrt(N)) for each test case, due to the loop.
 * Space Complexity: O(1).
 */
/***************************************************
 * Problem Name : C.cpp
 * Problem Link : https://codeforces.com/contest/1426/problem/C
 * OJ           : Codeforces
 * Verdict      : AC
 * Date         : 2020-10-01
 * Problem Type : Div/Edu -
 * Author Name  : Saikat Sharma
 * University   : CSE, MBSTU
 ***************************************************/
#include <algorithm>
#include <bitset>
#include <cassert>
#include <chrono>
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

#define __FastIO                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0)
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
#define nl cout << "\n"
#define sp cout << " "
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
#define un_map unordered_map
#define un_set unordered_set
#define pb push_back
#define mk make_pair
#define F first
#define S second
#define MAX 100005
#define INF 1000000009
#define MOD 1000000007

template <class T>
using min_heap = priority_queue<T, std::vector<T>, std::greater<T>>;
template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

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
struct custom_hash {
  static uint64_t splitmix64(uint64_t x) {
    // http://xorshift.di.unimi.it/splitmix64.c
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x + FIXED_RANDOM);
  }
};
ll lcm(ll a, ll b) { return (a / __gcd(a, b)) * b; }
/************************************ Code Start Here
 * ******************************************************/

// The following two functions are not used in the final solution.
ll fun(ll n) { return (n * (n + 1LL)) / 2LL; }

int bi(int n) {
  int low = 1, high = n;
  int ans = -1;

  while (high >= low) {
    int mid = (high + low) / 2;
    ll val = fun(mid);

    if (val >= n) {
      ans = mid;
      high = mid - 1;

    } else {
      low = mid + 1;
    }
  }

  return ans;
}

int main() {
  //~ __FastIO;
  //~ cout << setprecision (10) << fixed;
  int tc;
  cin >> tc;

  while (tc--) {
    int n;
    cin >> n;
    int ans = 1e9;

    // Iterate through possible numbers of "increase" moves.
    // Let x-1 be the number of "increase" moves. This results in x elements.
    // The number of copy moves needed is ceil((n-x)/x).
    for (int x = 1; x * x <= n; ++x) {
      ans = min(ans, x - 1 + ((n - x) + x - 1) / x);
    }

    cout << ans << endl;
  }

  return 0;
}
