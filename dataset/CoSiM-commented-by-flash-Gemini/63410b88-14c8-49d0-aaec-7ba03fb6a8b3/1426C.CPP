/**
 * @file 1426C.CPP
 * @brief Solution to Codeforces Problem 1426C: "Minimizing the maximum number of blocks".
 *
 * Algorithm: The problem asks for the minimum number of moves to reach a target number 'n' starting from '1'.
 * A move can either be incrementing the current number by 1, or multiplying the current number by some integer 'k'.
 * The solution explores all possible intermediate numbers 'x' that can be reached by only incrementing (cost `x-1`).
 * From 'x', to reach 'n', we need `ceil(n/x)` multiplications (or `ceil(n/x) - 1` "add x" operations and one initial `x`).
 * The total operations would be `(x - 1)` (to reach `x`) + `(ceil(n/x) - 1)` (to reach `n` from `x` by multiplying or repeated additions of `x`).
 * The formula used, `x - 1 + ((n - x) + x - 1) / x`, simplifies to `x - 1 + ceil((n - 1) / x)`, which effectively
 * represents the total moves to reach 'n'. The loop iterates `x` up to `sqrt(n)` because the function `f(x) = x-1 + (n-1)/x`
 * is minimized when `x` is close to `sqrt(n-1)`.
 *
 * Time Complexity: O(TC * sqrt(N)), where TC is the number of test cases and N is the input number.
 * Space Complexity: O(1).
 * Domain: Competitive Programming, Optimization, Number Theory.
 * Problem Link: https://codeforces.com/contest/1426/problem/C
 * OJ: Codeforces
 * Verdict: AC
 * Date: 2020-10-01
 * Problem Type: Div/Edu -
 * Author Name: Saikat Sharma
 * University: CSE, MBSTU
 */
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
// Functional Utility: Enables the use of GNU Policy-Based Data Structures (PBDS)
// for advanced data structures like ordered_set.
using namespace __gnu_pbds;
typedef long long ll;
typedef unsigned long long ull;

/**
 * Functional Utility: Optimizes C++ standard input/output streams by decoupling
 * them from C standard streams and disabling synchronization. This significantly
 * speeds up I/O operations, crucial for competitive programming performance.
 */
#define __FastIO                                                               \
  ios_base::sync_with_stdio(false);                                            \
  cin.tie(0);                                                                  \
  cout.tie(0)
// Functional Utility: Redirects standard input from "input.txt".
#define __FileRead freopen("input.txt", "r", stdin)
// Functional Utility: Redirects standard output to "output.txt".
#define __FileWrite freopen("output.txt", "w", stdout)
// Functional Utility: Sets all bytes of an array 'a' to a value 'v'.
#define SET(a, v) memset(a, v, sizeof(a))
// Functional Utility: Returns the size of a container 'v'.
#define SZ(v) (int)v.size()
// Functional Utility: Defines a shorthand for pair of integers.
#define pii pair<int, int>
// Functional Utility: Defines a shorthand for pair of int and long long.
#define pil pair<int, ll>
// Functional Utility: Defines a shorthand for pair of long long and int.
#define pli pair<ll, int>
// Functional Utility: Defines a shorthand for pair of long long.
#define pll pair<ll, ll>
// Functional Utility: Prints a debug separator to stdout.
#define debug cout << "######\n"
// Functional Utility: Prints a debug message with a single variable.
#define debug1(x) cout << "### " << x << " ###\n"
// Functional Utility: Prints a debug message with two variables.
#define debug2(x, y) cout << "# " << x << " : " << y << " #\n"
// Functional Utility: Prints a newline character.
#define nl cout << "\n";
// Functional Utility: Prints a space character.
#define sp cout << " ";
// Functional Utility: Scans a long long integer.
#define sl(n) scanf("%lld", &n)
// Functional Utility: Scans a double-precision floating-point number.
#define sf(n) scanf("%lf", &n)
// Functional Utility: Scans an integer.
#define si(n) scanf("%d", &n)
// Functional Utility: Scans a string.
#define ss(n) scanf("%s", n)
// Functional Utility: Scans an integer. (Typo: should be printf or similar if output)
#define pf(n) scanf("%d", n)
// Functional Utility: Scans a long long integer. (Typo: should be printf or similar if output)
#define pfl(n) scanf("%lld", n)
// Functional Utility: Returns iterators to the beginning and end of a container.
#define all(v) v.begin(), v.end()
// Functional Utility: Returns reverse iterators to the beginning and end of a container.
#define rall(v) v.begin(), v.end()
// Functional Utility: Sorts a container in ascending order.
#define srt(v) sort(v.begin(), v.end())
// Functional Utility: Sorts a container in descending order.
#define r_srt(v) sort(v.rbegin(), v.rend())
// Functional Utility: Reverses the order of elements in a container.
#define rev(v) reverse(v.rbegin(), v.rend())
// Functional Utility: Calculates the square of 'x'.
#define Sqr(x) ((x) * (x))
// Functional Utility: Calculates the modulo of 'x' with 'm', handling negative results correctly.
#define Mod(x, m) ((((x) % (m)) + (m)) % (m))
// Functional Utility: Returns the maximum of three values.
#define max3(a, b, c) max(a, max(b, c))
// Functional Utility: Returns the minimum of three values.
#define min3(a, b, c) min(a, min(b, c))
// Functional Utility: Defines a shorthand for unordered_map.
#define un_map unordered_map
// Functional Utility: Defines a shorthand for unordered_set.
#define un_set unordered_set
// Functional Utility: Pushes an element to the back of a container.
#define pb push_back
// Functional Utility: Creates a pair.
#define mk make_pair
// Functional Utility: Accesses the first element of a pair.
#define F first
// Functional Utility: Accesses the second element of a pair.
#define S second
// Functional Utility: Defines a constant for maximum array size or similar limits.
#define MAX 100005
// Functional Utility: Defines a constant for a large integer value, often used as infinity.
#define INF 1000000009
// Functional Utility: Defines a constant for a modulo value, common in problems involving large numbers.
#define MOD 1000000007

/**
 * @brief Custom min-heap (priority queue) template.
 * @tparam T The type of elements in the heap.
 * Functional Utility: Provides a standard library priority queue configured as a min-heap.
 */
template <class T>
using min_heap = priority_queue<T, std::vector<T>, std::greater<T>>;
/**
 * @brief Custom ordered_set template using policy-based data structures.
 * @tparam T The type of elements in the set.
 * Functional Utility: Provides a balanced binary search tree (red-black tree) that
 * allows for order statistics operations (e.g., finding k-th element, count elements less than x).
 */
template <typename T>
using ordered_set =
    tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;

/**
 * @brief Converts any type `T` to its string representation.
 * @tparam T The type to convert.
 * @param Number The value to convert.
 * @return String representation of the number.
 */
template <typename T> string toString(T Number) {
  stringstream ss;
  ss << Number;
  return ss.str();
}
/**
 * @brief Converts a string `str` to its integer representation.
 * @tparam T The type of the string (e.g., `std::string`).
 * @param str The string to convert.
 * @return Integer representation of the string.
 */
template <typename T> int toInt(T str) {
  stringstream ss;
  ss << str;
  int num;
  ss >> num;
  return num;
}
/**
 * @brief Custom hash function for `uint64_t` to be used with `unordered_map` or `unordered_set`.
 * Functional Utility: Provides a robust hash function (splitmix64) to reduce collisions,
 * especially useful in competitive programming for performance with hash-based containers.
 */
struct custom_hash {
  static uint64_t splitmix64(uint64_t x) {
    // http://xorshift.di.unimi.it/splitmix64.c
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
  }

  size_t operator()(uint64_t x) const {
    // Inline: Incorporates current time into the hash to provide a more random seed,
    // further reducing the chance of hash collisions for different program runs.
    static const uint64_t FIXED_RANDOM =
        chrono::steady_clock::now().time_since_epoch().count();
    return splitmix64(x + FIXED_RANDOM);
  }
};
/**
 * @brief Calculates the least common multiple (LCM) of two long long integers.
 * @param a The first number.
 * @param b The second number.
 * @return The LCM of 'a' and 'b'.
 */
ll lcm(ll a, ll b) { return (a / __gcd(a, b)) * b; }
/************************************ Code Start Here
 * ******************************************************/

/**
 * @brief Calculates the sum of the first 'n' natural numbers (1 + 2 + ... + n).
 * @param n The upper limit of the sum.
 * @return The sum of natural numbers up to 'n'.
 * Algorithm: Arithmetic series sum formula: n * (n + 1) / 2.
 */
ll fun(ll n) { return (n * (n + 1LL)) / 2LL; }

/**
 * @brief Performs a binary search to find the smallest integer 'mid' such that
 *        the sum of natural numbers up to 'mid' (`fun(mid)`) is greater than or equal to 'n'.
 *        This function is defined but currently not used in the main logic.
 * @param n The target sum.
 * @return The smallest 'mid' satisfying the condition, or -1 if no such 'mid' is found (conceptually).
 * Algorithm: Binary Search.
 * Time Complexity: O(log N).
 */
int bi(int n) {
  int low = 1, high = n;
  int ans = -1;

  // Block Logic: Iterates to find the smallest 'mid' using binary search.
  // Invariant: `ans` stores the best 'mid' found so far that satisfies the condition.
  while (high >= low) {
    int mid = (high + low) / 2;
    ll val = fun(mid);

    // Block Logic: If `fun(mid)` is sufficient, try for a smaller 'mid'.
    if (val >= n) {
      ans = mid;
      high = mid - 1;

    } else { // Block Logic: If `fun(mid)` is too small, increase 'mid'.
      low = mid + 1;
    }
  }

  return ans;
}

int main() {
  //~ __FastIO; // Functional Utility: Commented out for competitive programming templates, usually enabled.
  //~ cout << setprecision (10) << fixed; // Functional Utility: Commented out, typically used for floating-point output precision.
  int tc;
  // Block Logic: Reads the number of test cases.
  cin >> tc;

  // Block Logic: Iterates through each test case.
  while (tc--) {
    int n;
    // Block Logic: Reads the target number 'n' for the current test case.
    cin >> n;
    // Functional Utility: Initializes 'ans' with a very large value to find the minimum.
    int ans = 1e9;

    // Block Logic: Iterates through possible intermediate values 'x' from 1 up to sqrt(n).
    // This optimization is based on minimizing the function f(x) = (x-1) + ceil((n-1)/x),
    // which occurs when x is close to sqrt(n).
    // Invariant: `ans` always holds the minimum number of moves found so far to reach 'n'.
    for (int x = 1; x * x <= n; ++x) {
      // Functional Utility: Calculates the number of moves for the current 'x'.
      // x - 1: Moves to reach 'x' by incrementing from 1.
      // ((n - x) + x - 1) / x: Equivalent to ceil((n-1)/x). This represents the
      // number of "multiplication" or "add x" operations needed to reach 'n' from 'x'.
      ans = min(ans, x - 1 + ((n - x) + x - 1) / x);
    }

    // Block Logic: Outputs the minimum number of moves required for the current 'n'.
    cout << ans << endl;
  }

  return 0;
}
