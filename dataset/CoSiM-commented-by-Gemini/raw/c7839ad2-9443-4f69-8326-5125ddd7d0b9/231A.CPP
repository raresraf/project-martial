/**
 * @file raw/c7839ad2-9443-4f69-8326-5125ddd7d0b9/231A.CPP
 * @brief Solves the "Team" problem from a programming contest.
 *
 * This program calculates the number of problems a group of teams will solve.
 * A team of three will solve a problem if at least two of its members are
 * confident in the solution.
 *
 * @algorithm
 * A simple brute-force approach is used. The program iterates through each of
 * the `n` teams. For each team, it reads the confidence levels (1 for confident,
 * 0 for not) of the three members. It sums these values. If the sum is 2 or
 * greater, it means at least two members are confident, and a counter for
 * solved problems is incremented.
 *
 * @time_complexity O(N), where N is the number of teams/problems.
 * @space_complexity O(1).
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

// --- Competitive Programming Boilerplate ---
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

int n;

int main() {
  // Fast I/O setup
  ios::sync_with_stdio(0);
  cin.tie(0);
  cin >> n;
  int ans = 0; // Counter for problems that will be solved.
  /**
   * @block
   * Main loop iterating through each of the `n` problems/teams.
   */
  FOR(i, 1, n) {
    int a, b, c;
    cin >> a >> b >> c; // Read confidence levels of the three members.
    
    // If the sum of confidence levels is 2 or more, the team solves the problem.
    if (a + b + c >= 2)
      ans++;
  }
  cout << ans << endl;
  return 0;
}
