/**
 * @file 160A.CPP
 * @brief Solves a greedy problem: find minimum number of elements whose sum is greater than half of total sum.
 *
 * This program reads a series of test cases. For each test case, it reads 't' integers,
 * calculates their total sum, and then finds the minimum number of largest integers
 * whose sum exceeds half of the total sum. This is a classic greedy algorithm application.
 *
 * Algorithm: Greedy approach. Sort the numbers in descending order and iteratively
 *            add the largest numbers until their sum is greater than half of the total sum.
 * Time Complexity: O(T * N log N), where T is the number of test cases and N is the
 *                  number of integers in each test case. The N log N factor comes from sorting.
 * Space Complexity: O(N) for storing the integers for a single test case.
 */
#include <algorithm>
#include <iostream>
using namespace std;

int main() {
  /**
   * @brief Main function: Processes multiple test cases, each finding the minimum elements
   *        to exceed half the total sum.
   * @return 0 if the program executes successfully.
   */
  int t, num[101]; // 't': number of elements for current test case, 'num': array to store elements.
  /**
   * @brief Outer loop to process multiple test cases.
   * @pre Input stream provides 't' (number of elements) for each test case, ending with EOF.
   * @invariant For each iteration, a new test case is processed until EOF.
   */
  while (cin >> t && t != EOF) {
    int sum = 0, total = 0, Ans = 0; // 'sum': total sum of elements, 'total': sum of selected elements, 'Ans': count of selected elements.
    /**
     * @brief Reads 't' integers and calculates their total sum.
     * @pre 't' is the number of integers to read.
     * @post 'num' array is populated, 'sum' contains the total of all elements.
     * @invariant 'i' iterates from 0 to t-1.
     */
    for (int i = 0; i < t; i++) {
      cin >> num[i]; // Read each number.
      sum += num[i]; // Add to total sum.
    }
    // Functional Utility: Sorts the array 'num' in ascending order.
    // This is a prerequisite for the greedy approach, allowing us to pick largest elements first.
    sort(num, num + t);
    /**
     * @brief Greedy loop to select the minimum number of largest elements whose sum exceeds half the total sum.
     * @pre 'num' array is sorted in ascending order. 'sum' contains the total of all elements.
     * @post 'total' contains the sum of selected largest elements, 'Ans' contains their count.
     * @invariant 'i' iterates from the largest element downwards. Loop continues as long as 'total'
     *            is not greater than half of 'sum'.
     */
    for (int i = t - 1; i >= 0 && total <= sum / 2; i--) {
      total += num[i]; // Add the current largest number to 'total'.
      Ans++;           // Increment the count of selected numbers.
    }
    cout << Ans << endl; // Output the minimum count of numbers.
  }
  return 0;
}
