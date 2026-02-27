/**
 * @file 158A.CPP
 * @brief Solution for competitive programming problem: "Next Round".
 *
 * @details This program calculates the number of participants who will advance
 *          to the next round of a competition. Participants advance if their
 *          score is greater than or equal to the k-th place finisher's score,
 *          provided their score is strictly positive. This implementation uses
 *          a two-pass approach: first reading all scores into an array, then
 *          iterating to count the eligible participants.
 *          Algorithm: Two-pass iterative processing of scores.
 *          Time Complexity: O(N), where N is the number of participants, as scores
 *                           are read and processed in two separate linear passes.
 *          Domain: Competitive Programming.
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

using namespace std;

int main(void) {
  // Declare 'n' for the total number of participants and 'k' for the k-th place.
  int n, k;

  // Read the total number of participants (n) and the k-th place (k).
  cin >> n >> k;

  int counter = 0; // Initialize counter for advancing participants.
  // Declare a static-sized array 'a' to store scores. This array has a fixed size of 100,
  // which might be an issue if 'n' (number of participants) exceeds 100.
  int a[100];
  // Block Logic: Read all participant scores into the array 'a'.
  for (int i = 0; i < n; i++) {
    cin >> a[i]; // Read score for the current participant.
  }
  
  // Block Logic: Iterate through the stored scores to count participants who advance.
  // Invariant: 'counter' accumulates the count of eligible participants.
  for (int i = 0; i < n; i++) {
    // Precondition: A participant advances if their score is greater than or equal to
    //               the k-th place score (a[k-1]) AND their score is strictly positive.
    if (a[i] >= a[k - 1] && a[i] > 0)
      counter++; // Increment the counter for advancing participants.
  }

  cout << counter; // Output the final count of participants who advance.
  return 0;        // Indicate successful program termination.
}
