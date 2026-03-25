#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

/**
 * @file 231A.CPP
 * @brief Solution for the "Team" competitive programming problem.
 *
 * Problem Statement:
 * A team of three decides to solve a problem in a contest only if at least two
 * of them are sure about the solution. Given the confidence of each member
 * (1 for sure, 0 for not sure) for `n` problems, this program counts the
 * total number of problems the team will implement.
 *
 * Algorithm:
 * The program iterates through each problem, counts the number of confident
 * members, and if the count is two or more, it increments a solution counter.
 *
 * 1. Read the number of problems, `n`.
 * 2. Initialize a counter for solved problems, `ans`, to 0.
 * 3. Loop `n` times. In each iteration:
 *    a. Initialize a temporary counter for the current problem, `count`, to 0.
 *    b. Read the three confidence values.
 *    c. Increment `count` for each value that is 1.
 *    d. If `count` is 2 or greater, increment `ans`.
 * 4. Print the final value of `ans`.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */

// Macros are a common competitive programming technique for code brevity.
#define MX 100000
#define fs first
#define sec second
#define TC                                                                     \
  int T;
  cin >> T;
  while (T--)
#define loop(i, a, c) for (int i = 0; i < a; i++)
#define loop1(i, a, c) for (int i = 1; i <= a; i++)
#define pb(a) push_back(a)
#define all(a) (a).begin(), (a).end()
#define mp(x, y) make_pair((x), (y))
#define ll long long int

using namespace std;
int main() {
  int s; // Represents the number of problems to process.
  cin >> s;
  int ans = 0; // Counter for the number of problems that will be solved.

  // Loop through each of the 's' problems.
  while (s--) {
    int a, b, c;
    int count = 0; // Counter for confident members for the current problem.

    // Block Logic: Read opinions and count the number of '1's.
    cin >> a;
    if (a == 1)
      count++;
    cin >> b;
    if (b == 1)
      count++;
    cin >> c;
    if (c == 1)
      count++;
    
    // Pre-condition: Check if the team will solve the problem.
    if (count >= 2)
      ans++;
  }
  cout << ans;
}