/**
 * @file 158A.CPP
 * @brief Counts the number of participants who advance to the next round of a competition.
 *        Participants advance if their score is greater than or equal to the k-th place
 *        score AND their score is strictly positive (greater than or equal to 1).
 * Algorithm: Linear scan and comparison. Reads all scores, identifies the cutoff score
 *            (k-th score), then performs another linear scan to count participants
 *            meeting the two advancement criteria.
 * Time Complexity: O(N) where N is the number of participants, due to two linear passes
 *                  through the array.
 * Space Complexity: O(N) for storing the array of participant scores.
 */
#include <bits/stdc++.h> // Includes common standard library headers, such as iostream and vector.

using namespace std;

/**
 * @brief Main function to determine the number of advancing participants.
 * Functional Utility: Reads competition parameters and scores, then calculates
 *                     and prints the count of participants who qualify for the next round.
 */
int main() {
  int n;   // Total number of participants.
  int k;   // The k-th place defines the cutoff score.
  int val; // Stores the cutoff score (score of the k-th participant).
  int cnt = 0; // Counter for participants who advance to the next round.
  
  // Block Logic: Reads the total number of participants (n) and the k-th position.
  cin >> n >> k;

  // Declaration of a Variable Length Array (VLA). This is a C99 feature
  // and a GNU extension in C++, not standard C++. In standard C++, 'n'
  // would need to be a compile-time constant or a std::vector would be used.
  int arr[n]; 

  // Block Logic: Reads 'n' scores from standard input and stores them in the array.
  // Invariant: After this loop, 'arr' contains all 'n' participant scores.
  for (int i = 0; i < n; ++i) {
    cin >> arr[i];
  }

  // Block Logic: Determines the cutoff score for advancement.
  // Functional Utility: The k-th participant's score (`arr[k-1]`) sets the minimum
  //                     score required to advance. (k-1 because arrays are 0-indexed).
  val = arr[k - 1]; 

  // Block Logic: Iterates through all participant scores to count those who advance.
  // Pre-condition: 'val' has been correctly determined.
  // Invariant: 'cnt' accurately reflects the number of participants meeting the criteria.
  for (int i = 0; i < n; ++i) {
    // Functional Utility: Checks if a participant's score meets both advancement criteria:
    //                     1. Score is greater than or equal to the k-th score (`val`).
    //                     2. Score is strictly positive (greater than or equal to 1).
    if (arr[i] >= val && arr[i] >= 1) {
      cnt++; // Increment count if participant advances.
    }
  }

  // Functional Utility: Prints the final count of advancing participants.
  cout << cnt << endl;

  return 0; // Indicates successful program termination.
}
