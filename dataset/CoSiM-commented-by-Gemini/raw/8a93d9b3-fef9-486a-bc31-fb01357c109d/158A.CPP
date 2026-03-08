/**
 * @file 158A.CPP
 * @brief Counts participants advancing to the next round based on a score threshold.
 *
 * @details
 * This program solves the "Next Round" problem, where participants advance if
 * their score is positive and at least as high as the k-th place score.
 * This specific implementation relies on the problem's guarantee that the input
 * list of scores is sorted in non-increasing order. It iterates through the
 * scores and stops at the first participant who does not qualify. The final
 * loop index then represents the total number of participants who advanced.
 *
 * @section
 * Algorithm: Single-pass linear scan with early exit.
 * Time Complexity: O(N), for reading the input and the single pass for checking.
 * Space Complexity: O(50) -> O(1), due to the fixed-size array for storing scores.
 *
 * @note This solution uses a fixed-size array `arr[50]`, assuming 'n' will not exceed 50.
 */
#include <iostream>
using namespace std;

int main(void) {
  // n: total number of participants.
  // k: the rank of the participant establishing the score threshold.
  int n, k;
  // A fixed-size array to store scores. Assumes n <= 50.
  int arr[50];
  int i;

  cin >> n;
  cin >> k;

  // A guard condition to ensure k is a valid rank within n.
  if (n >= k) {
    // Read all n participant scores into the array.
    for (i = 0; i < n; i++)
      cin >> arr[i];

    /**
     * @brief This loop determines the number of qualifying participants.
     *
     * Logic: It iterates from the first participant, breaking as soon as it finds
     * a participant who does not meet the criteria. The criteria for failing are:
     * 1. The score is less than the k-th participant's score (arr[k-1]).
     * 2. The score is not positive (i.e., <= 0).
     * This logic correctly counts the winners only because the input scores are
     * guaranteed to be in non-increasing order.
     */
    for (i = 0; i < n; i++)
      if (arr[i] < arr[k - 1] || arr[i] <= 0)
        break; // Exit the loop at the first non-qualifying participant.

    // The value of 'i' after the loop equals the number of participants who passed the checks.
    cout << i << endl;
  }
}
