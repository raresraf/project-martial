/*
        SubmissionId	:	6511499
        ContestId	:	96
        Index	:	A
        ProblemName	:	Football
        ProblemTags	:	['implementation', 'strings']
        ProgrammingLanguage	:	GNU C++0x
        Verdict	:	OK
*/

/**
 * @file 96A.CPP
 * @brief Solves the "Football" problem from Codeforces.
 * @details The problem asks to determine if a situation is "dangerous", which
 * is defined as having seven or more consecutive players of the same team.
 * The teams are represented by '0' and '1'.
 *
 * @algorithm
 * The algorithm iterates through the string to find consecutive runs of identical
 * characters.
 * 1. Read the input string representing the line of players.
 * 2. Append a sentinel character (e.g., '#') to the end of the string. This
 *    simplifies the loop by ensuring the last run of players is also checked.
 * 3. Use a variable `idx` to keep track of the end of the last run of
 *    identical characters. Initialize it to -1.
 * 4. Iterate through the string from the first character up to the character
 *    before the sentinel.
 * 5. At each position `i`, if the character `s[i]` is different from the next
 *    character `s[i+1]`, it marks the end of a run.
 * 6. The length of this run is calculated as `i - idx`. If this length is 7
 *    or more, the situation is dangerous. Print "YES" and terminate.
 * 7. If a run ends, update `idx` to `i` to mark the end of the current run.
 * 8. If the loop finishes without finding any dangerous situation, print "NO".
 *
 * @complexity
 * Time Complexity: O(L), where L is the length of the input string, as it
 * involves a single pass through the string.
 * Space Complexity: O(L) to store the input string.
 */
#include <bits/stdc++.h>
using namespace std;

// A collection of common macros for competitive programming.
typedef long long ll;
typedef pair<int, int> pii;
#define X first
#define Y second
#define pb push_back
#define SQR(x) ((x) * (x))
#define SZ(x) ((int)((x).size())) // Gets the size of a container.
#define ALL(x) (x).begin(), (x).end()
#define debug(x) cerr << #x << " = " << x << endl
#define FOR(i, a, n) for (__typeof(n) i = (a); i <= (n); i++) // A flexible for-loop.
#define FORD(i, n, a) for (__typeof(n) i = (n); i >= (a); i--)
#define INF (1000 * 1000 * 1000)
#define MAXN 100010

string s; // The input string of '0's and '1's.

int main() {
  // Fast I/O for competitive programming.
  ios::sync_with_stdio(0);
  cin.tie(0);
  
  cin >> s;
  // Append a sentinel character to handle the last run of players easily.
  s += '#';
  
  // `idx` stores the index of the last character of the previous run.
  int idx = -1;
  
  /**
   * @block
   * @brief Iterate through the string to find the end of each consecutive run.
   */
  FOR(i, 0, SZ(s) - 2)
  if (s[i] != s[i + 1]) {
    // `s[i]` is the last character of a run of identical players.
    // The length of this run is `i - idx`.
    if (i - idx >= 7) {
      cout << "YES";
      return 0; // Dangerous situation found, exit immediately.
    }
    // The current run has ended. Update `idx` for the next run.
    idx = i;
  }
  
  // If the loop completes, no run of 7 or more was found.
  cout << "NO";
  return 0;
}
