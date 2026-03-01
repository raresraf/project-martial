/**
 * @file 160A.CPP
 * @author Ds (ds303077135@gmail.com)
 * @brief Solves an obscure optimization problem.
 *
 * @problem The specific problem this code solves is difficult to identify from the
 * code alone, as the logic does not match common competitive programming problems
 * with similar filenames (like 160A - "Twins").
 *
 * @algorithm The algorithm appears to be a greedy strategy for a kind of discount
 * or selection problem.
 * 1. It first reads 'm' numbers to find their minimum value, `r`.
 * 2. It then reads 'n' numbers into an array `a` and calculates their total sum, `tot`.
 * 3. The array `a` is sorted in ascending order.
 * 4. A `while` loop iterates backwards from the end of the sorted array `a`.
 *    In each step, it seems to process a group of `r+1` or `r+2` items.
 * 5. From the total sum `tot`, it subtracts one or two elements from within this
 *    group. The specific indices `a[pos-r]` and `a[pos-r-1]` suggest a complex
 *    selection rule.
 * 6. The final value of `tot` is printed.
 *
 * @time_complexity O(M + N log N), due to reading M items, sorting N items, and
 * a final pass over N items.
 * @space_complexity O(N) to store the array of 'n' items.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#define rep(i, n) for (int i = 1; i <= n; ++i)
using namespace std;

int m, n, q, r = 100001, a[100010];
long long tot;

int main() {
  // Read 'm' numbers and find the minimum value among them, stored in 'r'.
  scanf("%d", &m);
  rep(i, m) scanf("%d", &q), r = min(r, q);

  // Read 'n' numbers into array 'a' and calculate their total sum.
  scanf("%d", &n);
  rep(i, n) scanf("%d", &a[i]), tot += a[i];

  // Sort the 'n' numbers in ascending order.
  sort(a + 1, a + 1 + n);
  
  int pos = n;
  // This loop processes the sorted items from most expensive to least expensive.
  // The logic's purpose is unclear without the original problem statement.
  while (pos >= 1) {
    if (pos > r) {
      // Subtract an item based on the offset 'r'.
      tot -= a[pos - r];
      // Subtract a second item and advance the position pointer by r+2 or r+1.
      if (pos - r > 1)
        tot -= a[pos - r - 1], pos = pos - r - 2;
      else
        pos = pos - r - 1;
    } else
      break;
  }
  
  cout << tot << endl;
  return 0;
}
