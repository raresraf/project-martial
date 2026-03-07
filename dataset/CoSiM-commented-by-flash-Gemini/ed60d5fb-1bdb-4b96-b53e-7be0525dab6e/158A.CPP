/**
 * @file 158A.CPP
 * @brief Solution to a competitive programming problem involving counting participants
 * who have achieved a certain score or higher, up to a specified rank.
 * The program reads a total number of participants 'a' and a rank 'b'. It then
 * processes 'a' scores, counting how many are positive and also greater than or
 * equal to the score at rank 'b'. The loop may terminate early if conditions are not met.
 * Algorithm: Iterative array processing with early exit conditions.
 * Time Complexity: O(a), where 'a' is the number of participants.
 * Space Complexity: O(1), as only a few scalar variables are used.
 */
#include <iostream>
using namespace std;

int main() {
  // Declares 'a' for the total number of participants and 'b' for the target rank.
  int a, b;
  // 'cont' is a counter for qualified participants.
  int cont = 0;
  // Reads the total number of participants and the target rank.
  cin >> a >> b;
  
  // 'aux' stores the score of the previous participant for comparison.
  // 'aux2' stores the score of the current participant.
  int aux, aux2;
  
  /**
   * Block Logic: Iterates through each participant's score up to 'a' times.
   * Invariant: At the start of each iteration, 'i' represents the current participant's index (0-based).
   * Pre-condition: Input stream contains 'a' integers following 'a' and 'b'.
   * Post-condition: 'cont' holds the count of participants satisfying the criteria.
   */
  for (int i = 0; i < a; i++) {
    // Stores the current 'aux2' (previous participant's score) into 'aux'.
    aux = aux2;
    // Reads the score of the next participant into 'aux2'.
    cin >> aux2;
    
    /**
     * Conditional Logic: Checks if the current score is not positive.
     * If the current score is 0 or negative, no further participants can qualify
     * under the problem's rules, so the loop terminates.
     */
    if (!aux2 > 0) // Equivalent to (aux2 <= 0)
      break;
    
    /**
     * Conditional Logic: Checks for early termination based on rank and score comparison.
     * This condition applies after the 'b'-th participant. If the current participant
     * is beyond rank 'b' and their score is less than the previous participant's score,
     * it implies a decreasing score trend, and no further participants can qualify.
     */
    if ((i + 1) > b && aux2 < aux)
      break;
    
    // If all conditions are met, increments the counter for qualified participants.
    cont++;
  }
  // Prints the final count of qualified participants.
  cout << cont << endl;
  
  return 0;
}
