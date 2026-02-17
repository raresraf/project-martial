/**
 * @file 898B.CPP
 * @brief Solution for a problem equivalent to solving a linear Diophantine equation.
 * @details The program finds a pair of non-negative integers (x, y) that satisfy
 * the equation `a*x + b*y = n` for given positive integers `n`, `a`, and `b`.
 *
 * @algorithm Brute-force search over one variable.
 * @description The program iterates through all possible values for `x` from 0 up to `n/a`.
 * For each `x`, it calculates the corresponding `y = (n - a*x) / b`. It then
 * checks if this calculated `y` is a non-negative integer. The first solution
 * found is printed.
 * @complexity
 * - Time: O(n/a), as the loop iterates up to `n/a` times.
 * - Space: O(1).
 */
#include <bits/stdc++.h>

#define ll long long

using namespace std;

/**
 * @brief Unconventionally checks if an integer 'key' exists within a numerical
 * range [low, high].
 * @note This is a very inefficient way to check for existence in a continuous
 * integer range. A simple check `low <= key && key <= high` would suffice.
 * The function is not used on a sorted data structure where binary search is
 * typically applied.
 */
bool binary_search(ll low, ll high, ll key) {
  if (low <= high) {
    ll mid = low + (high - low) / 2;
    if (mid == key) {
      return true;
    } else if (mid > key) {
      return binary_search(low, mid - 1, key);
    } else {
      return binary_search(mid + 1, high, key);
    }
  } else {
    return false;
  }
}

/**
 * @brief Checks if a double is a whole number.
 */
bool isInteger(double a) { return floor(a) == a; }

int main() {
  ll n, a, b;
  cin >> n >> a >> b;
  bool flag = false;
  ll x, y;
  ll ans1, ans2;

  /**
   * @brief Iterates through possible values of x to find a solution.
   * The loop for `x` goes up to `n/a`, which is the maximum possible integer
   * value for `x` while ensuring `y` is not negative.
   */
  for (x = 0; x <= n / a; x++) {
    // Check if (n - a * x) is perfectly divisible by b.
    if (isInteger((n - a * x) * 1.0 / b)) {
      /**
       * @brief The following checks are redundant. The `isInteger` check already
       * confirms that `y` is an integer. The `binary_search` is an
       * unnecessarily complex way to confirm `y` is within the valid range
       * [0, n/b], which is already guaranteed by the loop condition on `x`.
       * The final check `ans1 * a + ans2 * b == n` is also mathematically
       * guaranteed by the calculation of `ans2`.
       */
      if (binary_search(0, n / b, (n - a * x) / b)) {
        flag = true;
        ans1 = x;
        ans2 = (n - a * x) / b;
        if (ans1 * a + ans2 * b == n) {
          break; // First solution found, exit loop.
        } else {
          continue;
        }
      }
    }
  }

  if (flag) {
    cout << "YES" << endl;
    cout << ans1 << " " << ans2 << endl;
  } else {
    cout << "NO" << endl;
  }

  return 0;
}