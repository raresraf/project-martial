/**
 * @5791afc8-6f8c-47df-b704-48b9aef84e33/158A.CPP
 * @brief This program determines the number of participants who qualify for the next round
 * based on their scores. It considers 'k' participants and a 'n' budget for selection,
 * applying specific rules for positive scores and ties in a non-increasing score sequence.
 *
 * Algorithm: Iterative selection through a score array with dynamic qualification rules and tie-breaking logic.
 * Time Complexity: O(k) where 'k' is the number of participants.
 * Space Complexity: O(k) for storing participant scores.
 */

#include <iostream> // For standard input/output operations (cin, cout).

using namespace std; // Uses the standard namespace to avoid prefixing std::.

int main() {
  int k, n; // Declares 'k' for the total number of participants and 'n' for the qualification budget/count.
  cin >> k >> n; // Reads 'k' and 'n' from standard input.
  
  // Declares an array 'a' to store the scores of 'k' participants plus one extra for a sentinel.
  int a[k + 1]; 
  /**
   * Block Logic: Reads 'k' participant scores into the array 'a'.
   * Invariant: After this loop, 'a[0]' to 'a[k-1]' contain participant scores,
   * assumed to be in non-increasing order based on problem context.
   */
  for (int i = 0; i < k; i++)
    cin >> a[i]; // Reads each score.
  
  // Sets a sentinel value at 'a[k]' (one past the last actual score).
  // This simplifies loop termination conditions by ensuring a[i+1] is always valid and 0 will act as a floor.
  a[k] = 0; 
  int i = 0; // Initializes 'i' as a counter for qualified participants, also used as an index.

  /**
   * Block Logic: Main loop to determine qualified participants.
   * It iterates through scores, counting participants as long as scores are non-increasing
   * and the qualification budget 'n' is available.
   * Invariant: 'i' represents the number of qualified participants found so far.
   * Pre-conditions: 'a' contains scores in non-increasing order.
   */
  while (a[i] >= a[i + 1] && n > 0) { // Continues as long as current score is >= next AND budget 'n' is not exhausted.
    // Inline: If the current score is 0, no further participants can qualify (assuming scores must be positive).
    if (a[i] == 0)
      break;
    n--; // Inline: Decrements the qualification budget 'n'.
    
    /**
     * Block Logic: Special handling when the qualification budget 'n' runs out.
     * If 'n' becomes 0, it means we've reached the target number of unique-score qualifiers.
     * However, all participants with the same score as the last qualified one should also be included.
     */
    if (n == 0) {
      // Inline: Increment 'i' to include all participants tied with the current score.
      while (a[i] == a[i + 1]) 
        i++;
    }
    i++; // Inline: Moves to the next participant.
  }
  cout << i << endl; // Prints the final count of qualified participants.
  return 0; // Indicates successful program termination.
}