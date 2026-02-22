/**
 * @90a223e2-50b6-42e9-842d-2df0a431f3d0/158A.CPP
 * @brief Solution for a competitive programming problem involving determining advancing participants based on scores.
 *
 * This program calculates the number of participants who advance to the next round of a competition.
 * A participant advances if their score is greater than or equal to the score of the k-th place finisher
 * and their score is strictly positive.
 *
 * Algorithm:
 * 1. Read the total number of participants (n) and the target rank (k).
 * 2. Read all 'n' participant scores into an array.
 * 3. Iterate through all participants' scores.
 * 4. For each participant, check if their score is both greater than or equal to the score
 *    of the participant at the k-th position (0-indexed k-1) and strictly greater than 0.
 * 5. Count the number of participants satisfying these conditions.
 * 6. Print the final count.
 *
 * Time Complexity: O(N) for reading input and a single pass to count advancing participants.
 *                  If sorting were enabled (commented out), it would be O(N log N).
 * Space Complexity: O(N) to store the participant scores.
 */

#include <algorithm> // Functional Utility: Provides generic algorithms, though std::sort is commented out here.
#include <iostream>  // Functional Utility: Provides input/output functionalities (std::cin, std::cout).
#include <vector>    // Functional Utility: Provides the std::vector container (not directly used, but a[n] uses dynamic array behavior in some compilers).
using namespace std;

int main() {
  int n, k;
  // Functional Utility: Reads the total number of participants 'n' and the target rank 'k'.
  cin >> n >> k;
  // Functional Utility: Declares a variable-length array (VLA) 'a' to store 'n' scores.
  int a[n];

  // Block Logic: Reads 'n' scores from the input and stores them in array 'a'.
  // Pre-condition: 'n' integer scores are available in the input stream.
  // Invariant: All 'n' scores will be read into the array.
  for (int i = 0; i < n; ++i) {
    cin >> a[i];
  }

  int ans = 0;
  // Functional Utility: The commented-out line indicates a potential alternative approach
  // involving sorting, which was not chosen for the current implementation.
  // sort(a, a+n);

  // Block Logic: Iterates through all participant scores to determine who advances.
  // Pre-condition: 'a' contains all participant scores; 'a[k-1]' holds the k-th score.
  // Invariant: 'ans' counts participants meeting the advancement criteria.
  for (int i = 0; i < n; ++i) {
    // Conditional Logic: Checks if the current participant's score meets the advancement criteria.
    // Criteria: Score is greater than or equal to the k-th place score AND strictly positive.
    if (a[i] >= a[k - 1] && a[i] > 0)
      ans++; // Functional Utility: Increments the count of advancing participants.
  }
  // Functional Utility: Prints the total number of advancing participants, followed by a newline.
  cout << ans << "\n";
}
