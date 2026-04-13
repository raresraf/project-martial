/**
 * @file 158A.CPP
 * @brief This program solves the "Next Round" problem. It counts how many participants
 * will advance to the next round of a competition. Participants advance if their score
 * is greater than or equal to the score of the participant who finished in the k-th place,
 * and their score is strictly positive.
 * Algorithm: Read scores, identify the k-th score, then count participants meeting criteria.
 * Time Complexity: O(N), where N is the number of participants, due to single pass for input and two passes for counting.
 * Space Complexity: O(N) for storing participant scores in an array.
 */
#include <cstdlib>  // Included in original, though not explicitly used in this solution.
#include <iostream> // Required for input/output operations (cin, cout).
using namespace std;

int main() {
  int n, k_place, p; // n: total participants, k_place: the k-th position (1-indexed), p: temporary variable for reading scores.
  int c = 0;         // 'c' is declared but not used in this implementation.

  // Reads the total number of participants 'n' and the k-th place 'k_place'.
  cin >> n >> k_place;

  // Declares a Variable Length Array (VLA) to store participant scores.
  // Note: VLAs are a C99 feature and a GNU extension in C++, not standard C++.
  int arr[n];

  /**
   * Block Logic: Reads 'n' participant scores from standard input and stores them in the 'arr' array.
   * Invariant: After this loop, 'arr' contains all participant scores in the order they were provided.
   */
  for (int i = 0; i < n; i++) {
    cin >> p;      // Reads a participant's score.
    arr[i] = p;    // Stores the score in the array.
  }

  // Adjusts 'k_place' to be 0-indexed for array access.
  k_place = k_place - 1;

  /**
   * Block Logic: Finds the effective k-th place participant, accounting for ties.
   * It extends 'k_place' towards higher indices as long as subsequent participants
   * have the same score as the current k-th place participant.
   * Invariant: 'k_place' points to the last participant who scored equal to the original k-th participant's score.
   */
  while (k_place + 1 < n) {
    if (arr[k_place] == arr[k_place + 1])
      k_place++; // Move to the next participant if scores are tied.
    else
      break; // Stop if scores differ.
  }
  /**
   * Block Logic: Adjusts 'k_place' to exclude participants with zero scores.
   * It moves 'k_place' backward towards lower indices as long as the participant's
   * score is 0. This ensures only positive scores are counted.
   * Invariant: 'k_place' points to the last valid participant (positive score) who qualifies.
   */
  while (k_place >= 0) {
    if (arr[k_place] == 0)
      k_place--; // Move to the previous participant if score is zero.
    else
      break; // Stop if score is positive.
  }

  // Prints the total count of participants advancing.
  // '+ 1' is added to convert the 0-indexed 'k_place' back to a 1-indexed count.
  cout << k_place + 1 << endl;
  // Standard program exit.
  return 0;
}
