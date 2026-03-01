/**
 * @file 1426C.CPP
 * @brief Solution to a number-theoretic optimization problem.
 * @problem The task is to find the minimum number of moves to reach at least a target
 * number 'n', starting from 1. Two types of moves are allowed:
 * 1. Increment the current number by one.
 * 2. Copy the current number and add it to the total (effectively, a multiplication).
 *
 * @algorithm The solution aims to minimize the total number of moves, which is the sum of
 * increment moves and copy moves. Let 'x' be the number obtained after some
 * initial increments. This takes `x-1` moves. To reach 'n' from 'x', we need
 * `ceil(n/x) - 1` copy operations. The total number of moves is `f(x) = (x - 1) + (ceil(n/x) - 1)`.
 * To minimize `f(x)`, the value of `x` should be close to `sqrt(n)`. The
 * code uses `x = ceil(sqrt(n))` as the optimal choice.
 *
 * @time_complexity O(1) for each test case, as it involves a constant number of arithmetic operations.
 * @space_complexity O(1)
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define N 300005
using namespace std;
typedef long long ll;
ll mod = 1000000007LL;
ll mod2 = 998244353LL;
int t, n, m;
int main() {
  cin >> t;
  // Loop through all test cases.
  for (int cas = 1; cas <= t; ++cas) {
    scanf("%d", &n);
    
    // Let 'i' be the value we reach by increments. The optimal choice for 'i'
    // is near sqrt(n). We choose i = ceil(sqrt(n)).
    ll i = (ll)sqrt(n);
    
    // Post-condition: if i*i is not exactly n, we need to increment i to get
    // the ceiling of the square root. This ensures i >= sqrt(n).
    if (i * i < n) {
      i++;
    }
    
    // Calculate the total moves.
    // (i - 1) moves are used to increment from 1 to 'i'.
    // (n + i - 1) / i is equivalent to ceil(n/i).
    // (ceil(n/i) - 1) moves are used for the copy-add operations.
    ll det = i - 1 + (n + i - 1) / i - 1;
    
    printf("%lld\n", det);
  }
  return 0;
}
