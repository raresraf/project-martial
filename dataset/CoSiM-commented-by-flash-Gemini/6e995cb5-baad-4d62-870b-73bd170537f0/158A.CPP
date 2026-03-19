/**
 * @file 158A.CPP
 * @brief Solution to Codeforces Problem 158A: "Next Round".
 * Counts the number of participants who will advance to the next round of a competition.
 * Participants advance if their score is positive and greater than or equal to the score
 * of the participant in the k-th place.
 * Algorithm: Read all scores, identify the k-th score as the threshold, then iterate
 * through all scores to count those meeting the criteria.
 * Time Complexity: O(N), where N is the number of participants. The `sort` operation
 *                   is redundant for correctness here but would add O(N log N) if it
 *                   were essential or if `pass` was determined after sorting.
 * Space Complexity: O(N) to store participant scores.
 */
#include <algorithm> // Required for std::sort.
#include <cstdio>    // Required for scanf and printf.
using namespace std;

/**
 * @brief Main function to solve the "Next Round" problem.
 * Reads participant scores, determines the k-th place score, and counts qualifiers.
 */
int main() {
  int n, k, pass; // 'n': total participants, 'k': the k-th place, 'pass': the minimum qualifying score.
  
  // Reads the total number of participants (n) and the k-th place (k).
  scanf("%d %d", &n, &k);
  
  // Declares a flexible array (C99 feature, or VLA in C++) to store 'n' scores.
  int x[n]; 
  int i; // Loop counter.

  /**
   * Block Logic: Reads 'n' scores from input and stores them in the array 'x'.
   * Pre-condition: 'n' is the number of scores to read.
   * Invariant: After each iteration, 'i+1' scores have been read into 'x'.
   */
  for (i = 0; i < n; i++) {
    scanf("%d", &x[i]);
  }

  // Determines the minimum score required to pass. This is the score of the k-th participant.
  // The problem statement guarantees scores are given in non-increasing order.
  pass = x[k - 1]; 
  int ans = 0; // Counter for the number of participants who advance.

  // NOTE: The sort operation here is redundant for the problem's logic because
  // the 'pass' score is determined from the original (non-increasingly sorted) input.
  // If 'pass' were to be determined after sorting, the logic would change.
  // For this specific problem and interpretation, this sort adds unnecessary overhead.
  sort(x, x + n); 

  /**
   * Block Logic: Iterates through the scores to count participants who meet the qualification criteria.
   * Criteria: Score must be positive (not zero) AND greater than or equal to the 'pass' score.
   * Invariant: 'ans' accumulates the count of qualified participants.
   */
  for (i = 0; i < n; i++) {
    // Checks if the current participant's score is positive and meets the passing threshold.
    if (x[i] != 0 && x[i] >= pass) {
      ans++; // Increments the counter if the participant qualifies.
    }
  }
  
  // Prints the total number of participants who advance to the next round.
  printf("%d", ans);
  return 0;
}
