/**
 * @file 158A.CPP
 * @brief Determines the number of participants who advance to the next round of a competition
 *        based on specific scoring and ranking rules, including edge cases for zero scores.
 *        Participants advance if their score is positive and greater than or equal to
 *        the score of the participant in the k-th position.
 * Algorithm: Linear scan and conditional counting. It first reads all scores, then applies
 *            different counting logic based on whether the leading score or the k-th score
 *            is zero, and finally sums the k qualified participants with any additional
 *            participants who tie the k-th score.
 * Time Complexity: O(N) where N is the number of participants, due to linear passes
 *                  through the array.
 * Space Complexity: O(N) for storing the array of participant scores.
 */
#include <iostream> // For standard input/output operations (cin, cout).

using namespace std;

// Type definitions for convenience, enhancing readability for specific integer types.
typedef long long int lli;  // Signed 64-bit integer.
typedef long long ll;       // Alias for signed 64-bit integer.
typedef unsigned long long int ulli; // Unsigned 64-bit integer.

/**
 * @brief Main function to calculate advancing participants.
 * Functional Utility: Reads competition parameters and participant scores,
 *                     then computes and prints the final count of qualifiers.
 * @param argc Number of command-line arguments (unused in this problem).
 * @param argv Array of command-line argument strings (unused in this problem).
 * @return 0 for successful execution.
 */
int main(int argc, char const *argv[]) {
  // Functional Utility: Optimizes C++ standard stream I/O by decoupling it from C stdio.
  //                     This can significantly speed up input/output operations in competitive programming.
  ios_base::sync_with_stdio(false);
  
  lli a; // Represents the total number of participants.
  lli b; // Represents the k-th position, defining the cutoff score.
  lli count = 0; // Counter for participants who meet the advancement criteria beyond the k-th position.
  
  // Block Logic: Reads the total number of participants (a) and the k-th position (b).
  cin >> a >> b;
  
  // Declaration of a Variable Length Array (VLA). This is a C99 feature
  // and a GNU extension in C++, not standard C++. In standard C++, 'a'
  // would need to be a compile-time constant or a std::vector would be used.
  lli arr[a]; 
  
  // Block Logic: Reads 'a' scores from standard input and stores them in the array.
  // Invariant: After this loop, 'arr' contains all 'a' participant scores.
  for (int i = 0; i < a; ++i) {
    cin >> arr[i];
  }
  
  // Block Logic: Handles the edge case where the highest score (or any score) is non-positive.
  // Functional Utility: If the first score is 0 or less, no one can advance, so the program exits early.
  if (arr[0] <= 0) {
    cout << 0 << endl; // Prints 0 as no one advances.
    return 0;          // Exits the program.
  }
  
  // Block Logic: Handles the edge case where the k-th score is zero.
  // Functional Utility: If the k-th participant's score is 0, only participants
  //                     with positive scores strictly *before* the k-th position advance.
  if (arr[b - 1] == 0) {
    // Counts participants with positive scores before the k-th position.
    for (int i = 0; i < b - 1; ++i) {
      if (arr[i] > 0) {
        count++;
      }
    }
    cout << count << endl; // Prints the count of advancing participants.
    return 0;              // Exits the program.
  }
  
  // Block Logic: Handles the general case where the k-th score is positive.
  // Functional Utility: Counts participants *after or at* the k-th position
  //                     whose scores are equal to the k-th score and are positive.
  // Invariant: 'count' will hold the number of ties with the k-th score beyond
  //            the k-th participant.
  for (int i = b; i < a; ++i) {
    if (arr[i] == arr[b - 1] && arr[i] > 0) {
      count++; // Increment count for participants who tie the k-th score and are positive.
    }
  }
  
  // Functional Utility: The total number of advancing participants is 'b' (all up to
  //                     k-th position, since arr[0]>0 and arr[b-1]>0 ensures they pass)
  //                     plus any additional participants who tied the k-th score.
  cout << b + count << endl;
  
  return 0; // Indicates successful program termination.
}
