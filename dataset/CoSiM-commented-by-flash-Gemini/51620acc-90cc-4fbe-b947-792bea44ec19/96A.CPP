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
 * @brief Solution to Codeforces Problem 96A: "Football".
 *        This program detects a "dangerous" situation in a football match
 *        represented by a string of '0's and '1's. A dangerous situation
 *        occurs if there are 7 or more consecutive players of the same team.
 *
 * Algorithm: Single-pass iteration with streak counting. It identifies continuous
 *            sequences of identical characters. `idx` keeps track of the start of the current streak.
 *            If a streak of 7 or more is found, it immediately outputs "YES".
 * Time Complexity: O(L), where L is the length of the input string, as it
 *                  involves a single pass through the string.
 * Space Complexity: O(L) for storing the input string, or O(1) if input is streamed.
 */

#include <bits/stdc++.h> // Includes various standard libraries for C++ competitive programming.
using namespace std;

// Competitive programming common practices:
typedef long long ll; // Defines 'll' as a shorthand for long long.
typedef pair<int, int> pii; // Defines 'pii' as a shorthand for pair<int, int>.
#define X first // Accesses the first element of a pair.
#define Y second // Accesses the second element of a pair.
#define pb push_back // Pushes an element to the back of a vector.
#define SQR(x) ((x) * (x)) // Calculates the square of a number.
#define SZ(x) ((int)((x).size())) // Gets the size of a container.
#define ALL(x) (x).begin(), (x).end() // Provides iterators for a container's full range.
#define debug(x) cerr << #x << " = " << x << endl // Debug macro to print variable name and value.
#define FOR(i, a, n) for (__typeof(n) i = (a); i <= (n); i++) // Loop macro (inclusive).
#define FORD(i, n, a) for (__typeof(n) i = (n); i >= (a); i--) // Loop macro (reverse inclusive).
#define INF (1000 * 1000 * 1000) // Defines a large integer as infinity.
#define MAXN 100010 // Defines a maximum value for N.

string s; // Global string variable to store the player arrangement.

int main() {
  /**
   * Functional Utility: Optimizes I/O and implements the streak detection logic.
   * Prints "YES" if a dangerous situation is found, otherwise "NO".
   */
  ios::sync_with_stdio(0); // Unties C++ streams from C standard streams for faster I/O.
  cin.tie(0);              // Unties cin from cout for faster input operations.

  cin >> s; // Reads the string representing player arrangements.
  s += '#'; // Appends a sentinel character to simplify boundary condition checks in the loop.

  int idx = -1; // `idx` stores the starting index of the current streak of identical characters.
                // Initialized to -1 to correctly handle a streak starting at index 0.

  /**
   * Block Logic: Iterates through the player arrangement string to find consecutive identical players.
   * Pre-condition: `s` contains the input string with a sentinel character appended. `idx` is -1.
   * Invariant: `idx` always points to the last character that broke a streak or the start of the string.
   *            `i - idx` represents the length of the current streak.
   */
  FOR(i, 0, SZ(s) - 2) { // Loop from the first character up to the second-to-last (to compare with i+1).
    // Condition Check: Detects a change in team or the end of the string (due to sentinel).
    if (s[i] != s[i + 1]) {
      // Condition Check: If the length of the current streak (`i - idx`) is 7 or more.
      if (i - idx >= 7) {
        cout << "YES"; // Output "YES" indicating a dangerous situation.
        return 0;      // Terminate the program immediately as the condition is met.
      }
      idx = i; // Reset `idx` to the current position, starting a new streak.
    }
  }

  cout << "NO"; // If the loop completes without finding a dangerous situation, output "NO".
  return 0;     // Indicates successful program termination.
}