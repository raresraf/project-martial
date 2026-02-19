/**
 * @file 231A.CPP
 * @brief Solves the "Team" problem from Codeforces.
 * @details This program calculates the number of problems a three-person team will
 * implement. The decision rule is that a problem is solved if at least two of the
 * three members are confident in the solution. This version uses several macros
 * to shorten the syntax.
 *
 * @algorithm The program iterates `n` times, once for each problem. In each loop,
 * it reads the three votes from the team members and sums them up. If the total
 * of 'sure' votes (represented by 1) is 2 or more, a result counter is incremented.
 * The final count is the answer.
 *
 * @time_complexity O(N), where N is the number of problems, due to the single loop
 * over the problems.
 * @space_complexity O(1), as it uses a constant amount of memory.
 */
// https://codeforces.com/contest/231/problem/A
#include <bits/stdc++.h>
using namespace std;
// Macros for abbreviating common I/O and loop constructs.
#define in cin >>
#define out cout <<
#define FOR(i, k, l) for (int i(k); i < l; i++)
#define Vi vector<int>
#define Msi map<string, int>
#define Mii map<int, int>

int main(void) {
  int n;
  in(n);
  // a, b, c: Represent the votes of the three team members.
  // res: The final count of problems to be implemented.
  int a, b, c, res = 0;

  /**
   * Block Logic: Iterate through each of the 'n' problems to be considered.
   */
  FOR(i, 0, n) {
    // s: A temporary sum of 'sure' votes for the current problem.
    int s = 0;
    in(a);
    in(b);
    in(c);

    // Tally the votes for the current problem.
    if (a == 1)
      s++;
    if (b == 1)
      s++;
    if (c == 1)
      s++;

    // Pre-condition: 's' holds the total number of members sure about the solution.
    // If the count is 2 or more, the team solves the problem.
    if (s >= 2)
      res++;
  }

  out(res);
}