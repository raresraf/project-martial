/**
 * @file 158A.CPP
 * @brief Solution for Codeforces Problem 158A: Next Round.
 * @details This program determines the number of participants who will advance to the next round of a competition.
 * A participant advances if their score is greater than or equal to the score of the k-th place finisher,
 * and their score is strictly positive. The program reads 'n' participants' scores, identifies the score
 * of the k-th place finisher, and then counts how many participants meet the advancement criteria.
 * @algorithm Iterative scanning and conditional counting.
 * Time Complexity: $O(N)$ where N is the number of participants.
 * Space Complexity: $O(N)$ to store participant scores.
 */
#include <iostream>
#include <vector>

using namespace std;

/**

 * @file 158A.CPP

 * @brief Solution for Codeforces Problem 158A: Next Round.

 * @details This program determines the number of participants who will advance to the next round of a competition.

 * A participant advances if their score is greater than or equal to the score of the k-th place finisher,

 * and their score is strictly positive. The program reads 'n' participants' scores, identifies the score

 * of the k-th place finisher, and then counts how many participants meet the advancement criteria.

 * @algorithm Iterative scanning and conditional counting.

 * Time Complexity: $O(N)$ where N is the number of participants.

 * Space Complexity: $O(N)$ to store participant scores.

 */

#include <iostream> // Required for input/output operations (std::cin, std::cout).

#include <vector>   // Required for dynamic array (std::vector).



using namespace std; // Using the standard namespace to avoid prefixing std::.



int main() {

  /**

   * @brief Main function: orchestrates the participant advancement logic.

   */

  int n,   // @var n: Total number of participants.

      k,   // @var k: The position of the participant whose score determines the cut-off.

      sum = 0; // @var sum: Counter for the number of participants who advance.



  cin >> n >> k; // Reads the total number of participants and the k-th position.

  

  // Functional Utility: Declares a dynamic array (vector) to store participant scores.

  // The size is 'n', and all elements are initialized to 0.

  vector<int> array1(n, 0); 



  /**

   * @brief Reads the scores of all participants.

   * @block_logic Iterates `n` times to read each participant's score.

   * @invariant `array1[i]` stores the score of the (i+1)-th participant.

   */

  for (int i = 0; i < n; i++) {

    cin >> array1[i]; // Reads the score for the current participant.

  }



  /**

   * @brief Determines and counts participants who meet the advancement criteria.

   * @block_logic Iterates through all participant scores and applies the advancement rules.

   * @invariant `sum` accumulates the count of participants whose score is sufficient and positive.

   */

  for (int i = 0; i < n; i++) {

    // Condition for advancement: score is greater than or equal to k-th place score AND score is positive.

    // Note: k-th place is (k-1) in 0-indexed array.

    if (array1[i] >= array1[k - 1] && array1[i] > 0) { 

      sum += 1; // Increments the counter if the participant advances.

    }

  }



  // Functional Utility: Prints the total number of participants who advance to the next round.

  cout << sum << endl;

  return 0; // Indicates successful execution.

}
