// Problem Link : http://codeforces.com/problemset/problem/1426/C

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

/**
 * @file 1426C.CPP
 * @brief Solution to the "Increase and Copy" problem from Codeforces (1426C).
 * @details This program calculates the minimum number of moves to obtain at least 'n' elements,
 * starting from one element. The allowed moves are: (1) incrementing the current number of
 * elements by one, and (2) copying all existing elements, effectively doubling the count achieved so far.
 *
 * Algorithm:
 * Let's say we perform `x-1` increment operations to get `x` elements. Then, we perform `y-1`
 * copy operations. The total number of elements becomes `x * y`. The total moves are `(x-1) + (y-1)`.
 * We need to minimize this sum, `x + y - 2`, subject to the constraint `x * y >= n`.
 * To minimize the sum of two numbers with a fixed product, the numbers should be as close as possible.
 * Thus, `x` and `y` should both be close to `sqrt(n)`.
 * The optimal strategy is to choose `x` as `floor(sqrt(n))`, perform `x-1` increment moves.
 * Then, the number of elements we need to reach is `n`, so we need `y` to be at least `ceil(n/x)`.
 * This requires `ceil(n/x) - 1` copy moves.
 * The total moves will be `(x - 1) + (ceil(n / x) - 1)`.
 *
 * The code implements this by setting `m = floor(sqrt(n))`. The number of increment moves is `c = m - 1`.
 * The number of copy moves is `ceil(n/m) - 1`.
 * `ceil(n/m)` is `(n/m)` if `n` is a multiple of `m`, and `floor(n/m) + 1` otherwise.
 * So, if `n % m == 0`, total moves = `(m-1) + (n/m) - 1`.
 * If `n % m != 0`, total moves = `(m-1) + (n/m) + 1 - 1 = (m-1) + (n/m)`.
 *
 * Time Complexity: O(1) for each test case, as it involves a constant number of arithmetic operations.
 * The `sqrt` function is highly optimized.
 * Space Complexity: O(1).
 */
int main() {
  ll t;
  cin >> t;
  // Loop for each test case.
  while (t--) {
    ll n, c = 0;
    cin >> n;
    // Let 'm' be the number of elements obtained after some increment operations.
    // To minimize moves, 'm' should be close to sqrt(n). We take the floor.
    ll m = sqrt(n);
    // 'c' is the number of increment moves to get 'm' elements (m-1 increments from 1).
    c = m - 1;

    // After 'm-1' increments, we have 'm' elements. Now we need to reach 'n'
    // using copy operations. Each copy operation creates a duplicate of the existing elements.
    // To reach 'n' from 'm', we need `ceil(n/m)` total "groups" of 'm'.
    // This requires `ceil(n/m) - 1` copy moves.
    if (n % m == 0)
      // If n is perfectly divisible by m, ceil(n/m) is n/m.
      // Total moves = (increments) + (copies) = (m-1) + (n/m - 1).
      cout << c + (n / m) - 1 << endl;
    else
      // If n is not perfectly divisible, ceil(n/m) is (n/m) + 1 in integer arithmetic.
      // Total moves = (increments) + (copies) = (m-1) + ((n/m) + 1 - 1) = c + (n/m).
      cout << c + (n / m) << endl;
  }
  return 0;
}
