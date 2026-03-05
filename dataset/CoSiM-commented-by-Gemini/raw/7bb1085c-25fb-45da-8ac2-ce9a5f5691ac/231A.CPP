//
//  main.cpp
//  231A
//
//  Created by Alejandro Gleason on 4/6/19.
//  Copyright   2019 Alejandro Gleason. All rights reserved.
//

/**
 * @file 231A.CPP
 * @brief Solution for Codeforces Problem 231A, "Team".
 * @details This program counts the number of problems a three-person team will
 * decide to implement. The criterion for implementation is that at least two
 * members must be sure about the solution.
 *
 * Algorithm:
 * 1. Read the total number of problems, `n`.
 * 2. Initialize a counter for the problems to be implemented to 0.
 * 3. Loop `n` times, once for each problem.
 * 4. In each iteration, read the three members' opinions (1 for sure, 0 for not).
 * 5. Sum the three opinions.
 * 6. If the sum is 2 or greater, increment the implementation counter.
 * 7. After the loop finishes, print the final count.
 *
 * Time Complexity: O(N), where N is the number of problems.
 * Space Complexity: O(1).
 */
#include <iostream>
using namespace std;

int main() {
  int n, total;
  int aux1, aux2, aux3, total2 = 0;
  cin >> n;
  /**
   * @brief This loop iterates through each of the `n` problems.
   */
  for (int i = 0; i < n; i++) {
    cin >> aux1 >> aux2 >> aux3;
    total = aux1 + aux2 + aux3;
    /**
     * @brief Checks if the problem meets the implementation criteria.
     * If the sum of opinions is 2 or more, it means at least two members
     * are confident.
     */
    if (total >= 2) {
      total2++;
    }
  }

  cout << total2;
  return 0;
}
