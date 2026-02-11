/**
 * @file 266A.CPP
 * @brief An alternative, inefficient solution to the "Stones on the Table" problem.
 * @details This program calculates the minimum number of stones to remove from a row
 * so that no two adjacent stones share the same color. It does so by iteratively
 * finding and removing a stone that is identical to one of its neighbors, counting
 * each removal. This approach is functionally correct but highly inefficient due to
 * repeated string manipulations.
 *
 * @see https://codeforces.com/problemset/problem/266/A
 *
 * Algorithm:
 * 1. Read the number of stones and their colors as a string.
 * 2. Iterate through the string, examining each stone (character).
 * 3. If a stone's color matches its left or right neighbor, increment a removal counter.
 * 4. Upon finding such a match, the stone is removed from the string. This involves
 *    creating a new, shorter string.
 * 5. The loop counter is then decremented to re-examine the character that has shifted
 *    into the current position.
 * 6. A special case handles a final two-stone string where both stones are the same color.
 *
 * Time Complexity: O(N^2) in the worst case. The string `substr` operation, which
 * creates a new string, takes O(N) time and is performed inside a loop that can also
 * run up to N times.
 * Space Complexity: O(N) for storing the string, but with significant memory churn due
 * to the creation of new string objects in each removal step.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <queue>
#include <stack>
#include <string>
#include <utility>
#include <vector>

// Macros for competitive programming, providing shortcuts for common operations.
#define fi first
#define se second
#define pb push_back
#define mp make_pair
#define pi 2 * acos(0.0)
#define eps 1e-9
#define PII pair<int, int>
#define PDD pair<double, double>
#define LL long long
#define INF 1000000000

using namespace std;

int len, ans;
// C-style char array for scanf.
char temp[100];
// C++ string for manipulation.
string str;

int main() {
  // Read length and then the string using C-style I/O.
  scanf("%d", &len);
  scanf("%s", temp);
  str = temp; // Convert to C++ string.

  ans = 0;

  /**
   * @brief Main loop to find and remove stones.
   *
   * Pre-condition: 'str' contains the initial sequence of stones.
   * Post-condition: 'ans' holds the total count of removed stones.
   * Logic: This loop is inefficient because it modifies the string it's iterating over.
   * Each `substr` call creates a new string, leading to poor performance.
   */
  for (int i = 1; i + 1 < str.size(); i++)
    if (str[i] == str[i - 1] || str[i] == str[i + 1]) {
      ans++;
      // Rebuild the string, removing the character at index 'i'.
      str = str.substr(0, i) + str.substr(i + 1);
      // Decrement 'i' to re-check the current index, as a new character has moved into this position.
      i--;
    }

  // Special case: If the loop reduces the string to two identical stones, one more removal is needed.
  if (str.size() == 2 && str[0] == str[1])
    ans++;

  printf("%d\n", ans);
  return 0;
}
