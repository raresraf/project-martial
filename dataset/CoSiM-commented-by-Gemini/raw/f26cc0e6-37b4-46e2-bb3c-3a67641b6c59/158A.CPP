/**
 * @file 158A.CPP
 * @brief Solves the "Next Round" problem by counting qualifying participants.
 *
 * This program determines how many participants will advance to the next round of a
 * competition. It reads the total number of participants, `n`, and the rank `k`.
 * The rule for advancing is that a participant's score must be positive and must be
 * equal to or greater than the score of the participant who finished in k-th place.
 * The program implements a single-pass solution to count the qualifiers.
 *
 * @author f26cc0e6-37b4-46e2-bb3c-3a67641b6c59
 *
 * @algorithm Single-Pass Iterative Counting
 * @complexity
 * Time: O(N), where N is the number of participants, as it iterates through the scores once.
 * Space: O(1), as it processes scores one by one without storing them all.
 */
#include <iostream>

int main() {
  // `n`: total number of participants.
  // `k`: the rank to compare against.
  int n, k;
  std::cin >> n >> k;
  // `count`: the number of participants who advance.
  // `val`: stores the score of the k-th participant.
  int count = 0, val;

  /**
   * @brief Loop to process each participant's score.
   * This loop iterates up to `n` times, but may break early based on the rules.
   * Pre-condition: `count` is 0.
   * Invariant: `count` is incremented for each participant that provisionally qualifies.
   */
  for (int i = 0; i < n; ++i) {
    int temp;
    std::cin >> temp;

    // A participant with a non-positive score cannot advance.
    // Since scores are non-increasing, no subsequent participant can advance either.
    if (temp <= 0)
      break;
    
    // When the k-th participant is reached (0-indexed i == k-1),
    // their score becomes the benchmark.
    else if (i == k - 1) {
      val = temp;
    } 
    // For participants after the k-th place...
    else if (i > k - 1) {
      // If their score is less than the k-th place score, they cannot advance.
      // The loop can terminate as no one else will qualify.
      if (temp < val)
        break;
    }
    // If none of the break conditions are met, the participant qualifies.
    ++count;
  }
  
  // Print the total number of qualified participants.
  std::cout << count << '
';
}