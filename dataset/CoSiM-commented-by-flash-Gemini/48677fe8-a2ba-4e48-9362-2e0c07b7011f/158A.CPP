/**
 * @48677fe8-a2ba-4e48-9362-2e0c07b7011f/158A.CPP
 * @brief This program counts the number of participants who advance to the next round
 *        based on a cutoff score determined by the K-th place finisher.
 * Algorithm: Linear scan to find the K-th score, followed by another linear scan to count eligible participants.
 * Time Complexity: O(N) where N is the number of participants, due to two passes through the array.
 * Space Complexity: O(N) for storing participant scores in an array.
 */
#include <iostream>

using namespace std;

int main() { // Corrected function signature: main should return an int.
  // Declare variables:
  // n: Total number of participants.
  // k: The k-th place to consider for the cutoff score.
  int n, k;
  // Functional Utility: Read the total number of participants (n) and the k-th place (k).
  cin >> n;
  cin >> k;
  // Declare an array to store the scores of 'n' participants.
  int arr[n];
  // i: Loop counter.
  // no: Counter for the number of participants who qualify.
  int i, no = 0;
  // Block Logic: Read the scores of all participants into the array.
  for (i = 0; i < n; i++)
    cin >> arr[i];
  // Functional Utility: Determine the cutoff score.
  // The cutoff score is the score of the participant at the (k-1)-th index (k-th place).
  int x = arr[k - 1];
  // Block Logic: Iterate through all participants to count those who qualify.
  // Pre-condition: A participant qualifies if their score is greater than or equal to the cutoff score 'x' AND their score is positive.
  for (i = 0; i < n; i++) {
    if (arr[i] >= x && arr[i] > 0)
      no++; // Increment the counter if the participant qualifies.
  }
  // Functional Utility: Output the total number of qualifying participants.
  cout << no << endl;
  return 0; // main function returns 0 to indicate successful execution.
}
