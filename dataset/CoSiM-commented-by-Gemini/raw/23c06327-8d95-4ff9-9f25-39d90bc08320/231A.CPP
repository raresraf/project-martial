//
//  231A.cpp
//  Codeforces
//
//  Created by Tico on 3/17/17.
//  Copyright   2017 Thiago Ribeiro. All rights reserved.
//

/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program determines how many problems a three-person team will solve.
 * The team solves a problem only if at least two members are sure about the solution.
 *
 * @algorithm Iterative counting based on a simple majority rule.
 * @time_complexity O(n), where 'n' is the number of problems. The loop runs 'n' times
 * with constant time operations inside.
 * @space_complexity O(1), as memory usage does not scale with the input size.
 */
#include <iostream>
#include <stdio.h>

using namespace std;

int main() {
  int n, x, y, z; // n: number of problems; x, y, z: confidence of team members
  int sum = 0, num = 0; // sum: temporary sum of confidence; num: count of problems to be solved

  cin >> n;

  /**
   * Block Logic: Iterates through each of the 'n' problems.
   * Invariant: At the start of each iteration, 'num' holds the count of problems
   * solved so far.
   */
  while (n--) {
    // Reads the confidence values (0 or 1) for the three members.
    scanf("%d %d %d", &x, &y, &z);
    sum += x + y + z;
    
    // Pre-condition: Check if the sum of confidence values meets the threshold of 2.
    if (sum >= 2)
      num++; // Increment the count of solvable problems.
      
    sum = 0; // Reset sum for the next problem.
  }

  // Output the final count of problems the team will solve.
  cout << num << endl;

  return 0;
}