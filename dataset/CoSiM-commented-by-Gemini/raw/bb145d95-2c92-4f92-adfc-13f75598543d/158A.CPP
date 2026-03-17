/**
 * @file 158A.CPP
 * @brief Solution to the "Next Round" problem.
 *
 * @details This program determines how many participants will advance to the next round of a contest.
 * A participant advances if their score is equal to or greater than the k-th place finisher's score,
 * and their score is positive. This implementation splits the logic into two main cases based on
 * whether the k-th participant's score is positive.
 *
 * @section
 * Algorithm:
 * 1. Read the number of participants 'n' and the rank 'k'.
 * 2. Read all 'n' scores into an array.
 * 3. Check if the score of the k-th participant is positive.
 *    - If yes, it assumes the first 'k' participants pass and then counts any additional
 *      participants who are tied with the k-th place score.
 *    - If no, it counts how many participants before the k-th place have a positive score.
 *
 * Time Complexity: O(N), for reading the scores and the subsequent linear scan.
 * Space Complexity: O(N), to store the scores in an array.
 */
/**
 * I'm sorry I was an Idiot
 */
#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int nums[n];
  // Reads all participant scores into an array.
  for (int i = 0; i < n; ++i) {
    cin >> nums[i];
  }

  /**
   * @brief This branch handles the case where the cutoff score (k-th place) is positive.
   */
  if (nums[k - 1] > 0) {
    int m = k;
    /**
     * @brief This loop extends the count of advancing participants beyond 'k'.
     * Logic: It finds how many participants after the k-th position have the exact same
     * score as the k-th participant, assuming they also advance.
     */
    while (m < n && nums[m] == nums[k - 1]) {
      ++m;
    }
    cout << m << endl;
  } 
  /**
   * @brief This branch handles the case where the cutoff score is zero or negative.
   * In this scenario, only participants with a score > 0 can advance.
   */
  else {
    int m = k - 2;
    /**
     * @brief This loop counts the number of participants who scored more than 0.
     * Logic: It iterates backward from the (k-1)-th position to find the last
     * participant with a positive score.
     */
    while (m >= 0 && nums[m] <= 0) {
      --m;
    }
    cout << m + 1 << endl;
  }
  return 0;
}