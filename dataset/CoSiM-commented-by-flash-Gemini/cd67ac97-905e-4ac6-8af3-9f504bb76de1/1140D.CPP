/**
 * @file 1140D.CPP
 * @brief This program calculates the sum of the product of consecutive integers
 *        from i=2 up to (a-1), where 'a' is a user-provided input.
 *        The sum is computed as: Sum[i * (i + 1)] for i from 2 to (a-1).
 * Algorithm: Direct Summation.
 * Time Complexity: O(A) where A is the input integer, due to a linear loop iteration.
 * Space Complexity: O(1) for constant variable storage.
 */
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

using namespace std;

// #include <ext/pb_ds/assoc_container.hpp>
// #include <ext/pb_ds/tree_policy.hpp>
// using namespace __gnu_pbds;
// typedef tree<int,null_type,less<int>,rb_tree_tag,
// tree_order_statistics_node_update> ordered_set;

int main(int argc, char const *argv[]) {
  int a;
  // Reads the upper limit 'a' for the summation.
  cin >> a;
  long long ans = 0;
  /**
   * Block Logic: Iterates from i=2 up to (a-1) to calculate the sum.
   * Invariant: In each iteration, 'ans' accumulates the product i * (i + 1).
   */
  for (int i = 2; i < a; ++i) {
    // Inline: Adds the product of the current integer 'i' and the next consecutive integer (i+1) to the total sum.
    ans += i * (i + 1);
  }
  // Prints the final calculated sum.
  cout << ans << endl;
  return 0;
}