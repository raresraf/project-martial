/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem. It counts how many problems
 *        a team of three friends will implement. A problem is implemented
 *        if at least two friends are confident in the solution (represented by 1).
 * Algorithm: Direct simulation/counting. It sums the confidence levels for each problem.
 * Time Complexity: O(N) where N is the number of problems, as it processes each problem once.
 * Space Complexity: O(1) for storing counters and problem-specific variables.
 */
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;                                                                       \
  cin >> T;                                                                    \
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int
using namespace std;
int main() {
  int s; // 's' represents the total number of problems to consider (conflicts with common usage of 's' for string).
  // Reads the total number of problems.
  cin >> s;
  int ans = 0; // Counter for problems that will be implemented.
  /**
   * Block Logic: Iterates 's' times, processing each problem.
   * Invariant: In each iteration, confidence levels for a problem are read,
   *            and 'ans' is updated if the problem meets the criteria.
   */
  while (s--) { // Loop continues 's' times, decrementing 's' each iteration.
    int a, b, c, count = 0; // a, b, c: confidence levels (0 or 1) of the three friends.
                            // count: number of friends confident in the current problem.
    cin >> a;               // Reads confidence of friend 1.
    if (a == 1)
      count++; // If friend 1 is confident, increment 'count'.
    cin >> b;  // Reads confidence of friend 2.
    if (b == 1)
      count++; // If friend 2 is confident, increment 'count'.
    cin >> c;  // Reads confidence of friend 3.
    if (c == 1)
      count++; // If friend 3 is confident, increment 'count'.
    if (count >= 2)
      ans++; // If at least two friends are confident, increment 'ans'.
  }
  // Prints the total number of problems that will be implemented.
  cout << ans;
}
