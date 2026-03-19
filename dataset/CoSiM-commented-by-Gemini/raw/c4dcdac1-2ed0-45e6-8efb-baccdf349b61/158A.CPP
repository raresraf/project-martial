/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem about qualifying scores.
 * @details This program determines how many contestants will advance to the next round.
 * A contestant advances if their score is positive and is greater than or equal to
 * the score of the k-th place contestant.
 *
 * @section Algorithm
 * 1. Read the total number of contestants (n) and the rank of the cutoff score (k).
 * 2. Read all 'n' scores into an array.
 * 3. Identify the score of the k-th ranked contestant (arr[k-1]).
 * 4. Iterate through all scores.
 * 5. Count how many scores are positive AND greater than or equal to the k-th score.
 * 6. Print the final count.
 *
 * @section Complexity
 * - Time Complexity: O(N), where N is the number of contestants, due to two separate linear scans of the scores array.
 * - Space Complexity: O(N), to store the scores of the N contestants.
 */
#include <iostream>
using namespace std;
int main() {
  // n: number of participants, k: the rank to check against.
  int n, k;
  cin >> n >> k;

  // Using a C-style variable-length array (VLA), which is a non-standard C++ extension.
  // A std::vector would be standard and safer.
  int arr[n];
  // This variable will store the score of the k-th place finisher.
  int temp;
  // Counter for participants who advance.
  int count = 0;

  /**
   * Block Logic: Read all participant scores into the array.
   * Pre-condition: 'n' is the number of scores to be read.
   */
  for (int i = 0; i < n; i++) {
    cin >> arr[i];
  }

  // Store the score of the k-th participant. This is the cutoff score.
  // Note: k is 1-based, array is 0-based, so we use k-1.
  temp = arr[k - 1];
  
  // A commented-out debug line, likely used to check the cutoff score.
  // cout<<endl<<temp<<endl;
  
  /**
   * Block Logic: Count participants who qualify.
   * Invariant: 'count' holds the number of qualified participants found so far.
   * A participant qualifies if their score is >= the cutoff score AND is positive.
   */
  for (int i = 0; i < n; i++) {
    if (arr[i] >= temp && arr[i] > 0) {
      count++;
    }
  }
  
  // Print the total number of qualified participants.
  cout << count << endl;

  return 0;
}
