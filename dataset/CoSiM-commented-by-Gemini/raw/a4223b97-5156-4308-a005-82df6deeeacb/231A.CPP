//
// Created by Taro Young on 2020/9/6.
//
/**
 * @file 231A.CPP
 * @brief Solution to the "Team" problem from Codeforces.
 *
 * This program calculates the number of problems a three-person team will implement.
 * The team decides to solve a problem if at least two members are confident
 * in the solution. The confidence of each member is given as a binary value (1 for sure, 0 for not sure).
 *
 * @algorithm Linear Scan/Counting
 * @time_complexity O(N), where N is the number of problems, as it processes each problem's data once.
 * @space_complexity O(1), as it only uses a few variables to store the state.
 */
#include <iostream>

using namespace std;

int main() {
  int n;
  int num = 0; // Counter for the number of problems to be implemented.
  cin >> n;

  /**
   * @brief Iterates through each problem to see if the team will solve it.
   *
   * For each of the `n` problems, it reads the confidence levels (P, V, T) of the three members.
   * If the sum of their confidence levels is 2 or greater, it means at least two members
   * agree, and the problem count is incremented.
   */
  for (int i = 0; i < n; i++) {
    int P, V, T;
    cin >> P >> V >> T;
    if (P + V + T >= 2)
      num++;
  }
  cout << num << endl;
  return 0;
}