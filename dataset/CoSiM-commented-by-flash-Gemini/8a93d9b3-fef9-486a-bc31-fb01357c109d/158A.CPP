/**
 * @8a93d9b3-fef9-486a-bc31-fb01357c109d/158A.CPP
 * @brief This program reads an array of 'n' integers and an integer 'k'.
 * It then counts how many elements in the array are greater than or equal to the
 * k-th element (1-indexed) and are also strictly positive. This is often used
 * in competitive programming to determine the number of participants who advance
 * in a contest based on a cutoff score.
 * Algorithm: Array traversal and comparison.
 * Time Complexity: O(N) for array input and another O(N) for counting, resulting in O(N).
 * Space Complexity: O(N) to store the input array.
 */

#include <iostream>
using namespace std;

int main(void) {
  // @var n: The total number of participants/elements in the array.
  // @var k: The 1-indexed position of the cutoff score.
  int n, k;
  // @var arr: An array to store the scores/elements. Max size 50.
  int arr[50];
  // @var i: Loop counter for iterating through the array.
  int i;

  // Block Logic: Read the total number of participants (n) and the cutoff rank (k).
  cin >> n;
  cin >> k;

  // Block Logic: Process only if the number of participants is at least the cutoff rank.
  // Pre-condition: 'n' must be greater than or equal to 'k'.
  if (n >= k) {
    // Block Logic: Read 'n' integer scores into the array.
    for (i = 0; i < n; i++)
      cin >> arr[i];

    // Block Logic: Iterate through the array to count eligible participants.
    // Invariant: 'i' counts the number of elements that satisfy the conditions.
    // Pre-condition: The array 'arr' is populated with 'n' scores.
    for (i = 0; i < n; i++)
      // Block Logic: Check if the current element is less than the k-th element's score
      // or if it is non-positive. If either is true, break the loop as subsequent elements
      // (assuming sorted, though not explicitly stated, this is typical competitive programming logic)
      // will also not meet the criteria, or if not sorted, it's a count of elements satisfying condition.
      if (arr[i] < arr[k - 1] || arr[i] <= 0)
        break;

    // Post-condition: Print the final count of eligible participants.
    // The loop counter 'i' at this point holds the count because the loop breaks
    // as soon as a non-qualifying participant is found or all participants are checked.
    cout << i << endl;
  }
  return 0; // Explicitly return 0 for successful execution.
}
