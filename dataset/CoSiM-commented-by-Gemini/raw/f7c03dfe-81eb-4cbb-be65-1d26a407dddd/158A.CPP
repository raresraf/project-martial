/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem (Codeforces 158A).
 *
 * Problem Statement:
 * Given the scores of 'n' participants in non-increasing order, determine how many
 * will advance to the next round. A participant advances if they have a positive
 * score and their score is equal to or greater than the score of the k-th place
 * finisher. 'k' is given as 'place'.
 *
 * Algorithm:
 * The code iterates through the 'contestant' list. It's designed to count
 * everyone up to the k-th 'place' who has a positive score. After the k-th
 * place, it continues to count participants only if they have the exact same
 * score as the k-th place finisher. The loop terminates early if a participant
 * after the k-th place has a lower score, which is an optimization based on the
 * non-increasing score order.
 *
 * Note on Implementation:
 * The variable 'lastscore' is used to hold the score of the k-th place finisher.
 * However, it is not initialized, which could lead to undefined behavior if the
 * first few contestants have a score of 0. The logic also doesn't explicitly
 * check for a positive score for participants after the k-th place, relying on
 * the k-th place score being positive, which might not always be the case.
 *
 * Time Complexity: O(n) in the worst case, as it might iterate through all contestants.
 * Space Complexity: O(1), using a fixed number of variables.
 */
#include <iostream>

using namespace std;

int main() {
  int contestant;
  int place;
  int counter = 0;
  int score;
  int lastscore; // Potential issue: uninitialized variable.
  
  // Read the number of contestants and the target rank (k).
  cin >> contestant;
  cin >> place;
  
  /**
   * Block Logic: Iterate through each contestant to count who advances.
   */
  for (int i = 0; i < contestant; i++) {
    cin >> score;
    
    // This condition handles participants up to the k-th place.
    if (i < place && score > 0) {
      counter++;
      // Store the score of the k-th place finisher.
      lastscore = score;
    } 
    // This condition handles participants after the k-th place.
    // They advance only if their score equals the k-th place score.
    else if (score == lastscore) {
      counter++;
    } 
    // Since scores are non-increasing, if a participant after k-th place
    // has a lower score, no one else can qualify.
    else {
      break;
    }
  }
  
  // Output the total number of participants who advance.
  cout << counter;
  return 0;
}
