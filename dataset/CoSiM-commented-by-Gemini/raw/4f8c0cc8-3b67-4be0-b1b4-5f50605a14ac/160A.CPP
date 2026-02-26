/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem with a greedy algorithm and custom input.
 *
 * @details The problem is to find the minimum number of coins to take from a
 * given set so that the sum of the selected coins is strictly greater than the
 * sum of the remaining coins.
 *
 * @algorithm
 * A greedy strategy is employed:
 * 1. Read all coin values, storing them in a vector and calculating their total sum.
 * 2. Sort the vector of coins in descending order.
 * 3. Iterate through the sorted coins, progressively taking the largest ones.
 *    In each step, add the coin's value to a `taken_sum` and subtract it from
 *    the `remaining_sum`.
 * 4. The loop continues as long as `taken_sum <= remaining_sum`.
 * 5. The number of coins taken when the loop terminates is the minimum required.
 *
 * @note This solution includes a custom `scanint` function for fast I/O, a
 * common micro-optimization in competitive programming.
 *
 * @complexity
 * Time Complexity: O(N log N), dominated by the sorting operation.
 * Space Complexity: O(N) to store the coin values in a vector.
 */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

// The `getchar_unlocked` is a non-standard, thread-unsafe version of getchar
// that can offer a performance boost by avoiding stream locking.
#define gc getchar_unlocked
inline int getchar_unlocked() { return getchar(); }
void scanint(int &x);

/**
 * @brief A custom function for fast integer input.
 * @details Reads character by character from stdin and manually constructs the
 * integer. This can be faster than `std::cin` or `scanf` but is less portable
 * and less safe.
 */
void scanint(int &x) {
  register int c = gc();
  x = 0;
  int neg = 0;
  for (; ((c < 48 || c > 57) && c != '-'); c = gc())
    ;
  if (c == '-') {
    neg = 1;
    c = gc();
  }
  // This loop builds the integer. `(x<<1) + (x<<3)` is a bit-shift
  // trick for `x*10` (2*x + 8*x).
  for (; c > 47 && c < 58; c = gc()) {
    x = (x << 1) + (x << 3) + c - 48;
  }
  if (neg)
    x = -x;
}

int main() {
  int n, i, j;
  int remaining_sum = 0; // Starts as the total sum, then decreases.
  int taken_sum = 0;     // Sum of coins taken.
  
  vector<int> a;
  scanint(n);

  // Read all coin values, storing them and calculating the total sum.
  for (i = 0; i < n; i++) {
    scanint(j);
    remaining_sum += j;
    a.push_back(j);
  }

  // Sort the vector in descending order using reverse iterators.
  sort(a.rbegin(), a.rend());
  
  i = 0; // `i` will now be used to count the number of coins taken.
  /**
   * @block
   * @brief Greedily take coins until the sum is greater than the remainder.
   * @invariant In each iteration, the largest remaining coin is taken.
   */
  while (taken_sum <= remaining_sum) {
    taken_sum += a[i];
    remaining_sum -= a[i];
    i++;
  }
  
  // `i` now holds the minimum number of coins needed.
  cout << i << endl;
  return 0;
}
