/**
 * @file 231A.CPP
 * @brief This program solves the "Team" problem. It counts how many problems
 *        a team of three friends will implement. A problem is implemented
 *        if at least two friends are confident in the solution.
 * Algorithm: Direct simulation/counting. Each problem's confidence levels are summed.
 * Time Complexity: O(N) where N is the number of problems, due to a single linear pass.
 * Space Complexity: O(1) for storing problem count and total implemented problems.
 */
/*
        SubmissionId	:	6511443
        ContestId	:	231
        Index	:	A
        ProblemName	:	Team
        ProblemTags	:	['brute force', 'greedy']
        ProgrammingLanguage	:	GNU C++0x
        Verdict	:	OK
*/

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<int, int> pii;
#define X first
#define Y second
#define pb push_back
#define SQR(x) ((x) * (x))
#define SZ(x) ((int)((x).size()))
#define ALL(x) (x).begin(), (x).end()
#define debug(x) cerr << #x << " = " << x << endl
#define FOR(i, a, n) for (__typeof(n) i = (a); i <= (n); i++)
#define FORD(i, n, a) for (__typeof(n) i = (n); i >= (a); i--)
#define INF (1000 * 1000 * 1000)
#define MAXN 100010

int n; // Declares 'n' as a global variable to store the number of problems.

int main() {
  // Functional Utility: Optimizes C++ standard input/output operations for faster execution.
  ios::sync_with_stdio(0); // Unties C++ standard streams from C standard streams.
  cin.tie(0);              // Unties cin from cout, preventing flushes.

  // Reads the total number of problems.
  cin >> n;
  int ans = 0; // Initializes 'ans' to count the number of problems that will be implemented.
  /**
   * Block Logic: Iterates through each of the 'n' problems, reading the confidence levels
   *              of the three friends and checking if at least two are confident.
   * Invariant: 'ans' accumulates the count of problems where the sum of confidence is 2 or more.
   */
  FOR(i, 1, n) { // Loop from 1 to n (inclusive) for each problem.
    int a, b, c; // Confidence levels of the three friends (1 if confident, 0 otherwise).
    cin >> a >> b >> c; // Reads confidence levels for the current problem.
    if (a + b + c >= 2) // Checks if at least two friends are confident (sum of 1s is >= 2).
      ans++;            // Increments the count of implementable problems.
  }
  // Prints the total number of problems that will be implemented.
  cout << ans << endl;
  return 0;
}