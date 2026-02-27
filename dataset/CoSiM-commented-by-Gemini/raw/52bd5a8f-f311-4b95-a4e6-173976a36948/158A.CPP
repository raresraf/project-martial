/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem from Codeforces.
 * @details The program counts how many participants will advance to the next round.
 * The rule for advancing is that a participant's score must be positive and
 * also greater than or equal to the score of the k-th place participant. The
 * input scores are provided in non-increasing order.
 *
 * @algorithm
 * 1. Read the number of participants `n` and the rank `k`.
 * 2. Read the `n` scores into an array.
 * 3. Determine the threshold score by getting the score of the k-th participant.
 * 4. Initialize a counter for advancing participants to 0.
 * 5. Iterate through the array of scores from the beginning.
 * 6. For each score, check if it is positive and greater than or equal to the
 *    threshold score.
 * 7. If it is, increment the counter.
 * 8. If a score does not meet the criteria, stop iterating immediately. Since the
 *    scores are in non-increasing order, no subsequent participant can qualify.
 * 9. Print the final counter value.
 *
 * @complexity
 * Time Complexity: O(n) for reading the input. The counting loop runs at most
 * n times, so the overall complexity is dominated by the input reading.
 * Space Complexity: O(n) to store the scores, although a fixed-size array
 * is used here.
 */
#include <algorithm>
#include <iostream>

#define MAX 52 // Defines a static max size for the array of scores.
using namespace std;

int main() {
  int n, k; // n: number of participants, k: rank of the threshold score.
  int arr[MAX]; // 1-indexed array to store participant scores.
  
  cin >> n >> k;
  
  // Read the scores into the array.
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }

  int value = arr[k]; // The score of the k-th participant is the threshold.
  int count = 0;      // Counter for participants who advance.

  /**
   * @block
   * @brief Count the number of participants who qualify for the next round.
   */
  for (int j = 1; j <= n; j++) {
    // A participant advances if their score is positive AND >= the k-th score.
    if (arr[j] >= value && arr[j] > 0)
      count++;
    else
      // Since scores are non-increasing, if one participant fails, all
      // subsequent participants will also fail. We can stop early.
      break;
  }

  cout << count << endl;
  return 0;
}
