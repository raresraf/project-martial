/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (Codeforces 158A).
 * @details The problem is to count how many participants will advance to the next round
 *          of a competition. A participant advances if they have a positive score
 *          and their score is greater than or equal to the score of the k-th place
 *          finisher.
 *
 * @algorithm The algorithm first reads all the scores to identify the score of the
 *            k-th place participant. This score serves as the benchmark. Then, it
 *            iterates through all participants a second time, counting how many
 *            have a positive score that meets or exceeds the benchmark score.
 *
 * @time_complexity O(N), where N is the number of participants, because it requires
 *                  two passes through the list of scores.
 * @space_complexity O(N) to store the scores of the participants.
 */
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, k;
  cin >> n >> k;
  // Use a Variable Length Array (VLA), a non-standard C++ extension.
  // Using 1-based indexing for the array to align with problem's 1-based k-th place.
  int arr[n + 1]; 
  for (int i = 1; i <= n; i++) {
    cin >> arr[i];
  }
  
  // The score of the k-th place finisher is the benchmark for advancing.
  int reqScore = arr[k];
  int count = 0; // Counter for participants who advance.

  /**
   * @brief Iterate through all participants to check if they advance.
   */
  for (int i = 1; i <= n; i++) {
    /**
     * @brief A participant advances if their score is positive AND greater than or
     * equal to the k-th participant's score.
     */
    if ((arr[i] > 0) && (arr[i] >= reqScore)) {
      count += 1;
    }
  }
  
  // Print the final count of participants who advance.
  cout << count << "\n";
  return 0;
}
