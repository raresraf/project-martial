/**
 * @file 231A.CPP
 * @brief Solution for a competitive programming problem (Codeforces "Team").
 * This program calculates the number of problems a three-person team will solve,
 * given that a problem is solved only if at least two members are confident.
 *
 * @author sumanth2303
 *
 * @algorithm Simulation with Online Processing. The program reads the data for
 * each problem one by one, decides if it will be solved, and updates a counter,
 * without storing the entire input.
 *
 * @time_complexity O(N), where N is the number of problems, as each problem is
 * processed once.
 * @space_complexity O(1), as data for each problem is processed on the fly.
 */
// problem link - https://codeforces.com/problemset/problem/231/A
// sumanth2303
// Team - 231A
// ACCEPTED
// One day three best friends Petya, Vasya and Tonya decided to form a team and
// take part in programming contest..
#include <bits/stdc++.h>
#include <iostream>
using namespace std;
int main() {
  int petya, vasya, tonya; // Confidence levels (0 or 1) for each member
  int n;                  // Total number of problems
  int score = 0;          // Counter for solvable problems
  cin >> n;

  /**
   * @block
   * @brief Processes each problem to see if it will be solved.
   *
   * The loop iterates `n` times. In each iteration, it reads the confidence
   * levels of the three members for a single problem. If the sum of their
   * confidence is 2 or more, the problem is counted as solved.
   */
  while (n--) {
    cin >> petya >> vasya >> tonya;
    if (petya + vasya + tonya >= 2) {
      score++;
    }
  }
  cout << score << endl;
  return 0;
}
