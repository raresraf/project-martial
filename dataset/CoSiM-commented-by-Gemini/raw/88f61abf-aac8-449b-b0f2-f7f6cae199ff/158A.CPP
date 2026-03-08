/**
 * @file 158A.CPP
 * @brief Solution to a problem determining the number of contestants advancing to the next round.
 *
 * @details
 * This program calculates how many contestants will advance based on their scores.
 * The rule for advancing is that a contestant's score must be positive and
 * greater than or equal to the score of the k-th ranked contestant. The program
 * reads the total number of contestants (n), the rank (k), and then all n scores.
 * It uses the k-th contestant's score as a threshold to count how many qualify.
 *
 * @section
 * Algorithm: Threshold comparison.
 * Time Complexity: O(N) due to two separate passes through the list of scores (one for input, one for counting).
 * Space Complexity: O(N) to store the scores of the n contestants in a dynamic array.
 */
#include <iostream>
using namespace std;

int main() {
  // n: total number of participants.
  // k: the rank of the participant establishing the score threshold.
  int n, k;
  cin >> n >> k;
  // Dynamically allocate an array to hold the scores.
  int *a = new int[n];
  int i;
  // Read all n participant scores into the array.
  for (i = 0; i < n; i++)
    cin >> a[i];

  // 'bar' is the threshold score, which is the score of the k-th participant (at index k-1).
  int bar = a[k - 1];
  // 'c' is the counter for the number of participants who advance.
  int c = 0;

  /**
   * @brief This loop iterates through all participants to count who qualifies.
   *
   * Logic: A participant qualifies if their score is positive AND greater than or equal
   * to the threshold score ('bar').
   * Invariant: 'c' holds the count of qualifying participants from the first 'i-1' scores.
   */
  for (i = 0; i < n; i++) {
    if (a[i] > 0 && a[i] >= bar)
      ++c;
  }

  // Output the final count of qualifying participants.
  cout << c << endl;

  /**
   * @brief Deallocates the memory used for the scores array.
   * @note The use of 'delete a;' is technically incorrect for an array allocated
   * with 'new int[n]'. The correct operator is 'delete[] a;'. However, in many
   * simple environments, this may not cause an immediate issue. The code is
   * preserved as-is.
   */
  delete a;
  a = NULL; // Setting the pointer to NULL after deletion is good practice to prevent dangling pointers.

  return 0;
}
