/**
 * @file 160A.CPP
 * @brief Finds the minimum number of largest elements in an array whose sum
 *        is strictly greater than half of the total sum of all elements.
 * Algorithm: Greedy approach. Sorts the array in non-decreasing order and then
 *            iterates from the largest elements (end of the sorted array)
 *            summing them up until their accumulated sum exceeds half of the
 *            total sum of the array.
 * Time Complexity: O(N log N) primarily due to the sorting step, where N is
 *                  the number of elements in the array. The subsequent iteration
 *                  is O(N).
 * Space Complexity: O(N) for storing the elements in the vector.
 */
#include <algorithm> // Required for std::sort.
#include <iostream>  // Required for standard input/output operations (cin, cout).
#include <stdlib.h>  // Included for exit(), though not used in this specific implementation.
#include <vector>    // Required for std::vector.

using namespace std;

/**
 * @brief Main function to solve the array partitioning problem.
 * Functional Utility: Reads a list of integers, determines the minimum count
 *                     of largest integers that sum to more than half the total,
 *                     and prints this count.
 */
int main() {
  int a, b;     // 'a' is used for the count of elements to read, 'b' for each element's value.
  int sum = 0;  // Stores the total sum of all elements in the vector.
  int sum2 = 0; // Stores the accumulated sum of the largest elements.
  int cnt = 0;  // Counts how many largest elements are taken.
  vector<int> ele; // Vector to store the input integers.
  
  // Block Logic: Reads the number of elements 'a', then reads 'a' integers
  //              into the 'ele' vector.
  cin >> a;
  while (a--) {
    cin >> b;
    ele.insert(ele.end(), b); // Inserts element 'b' at the end of the vector.
  }
  
  // Block Logic: Calculates the total sum of all elements in the vector.
  // Invariant: 'sum' accurately holds the sum of all elements.
  for (vector<int>::iterator it = ele.begin(); it != ele.end(); ++it) {
    sum += *it;
  }
  
  // Functional Utility: Sorts the vector 'ele' in non-decreasing order.
  // Pre-condition: 'ele' contains integers.
  sort(ele.begin(), ele.end());
  
  // for(vector<int>::iterator it=ele.begin(); it!=ele.end();++it) {cout<<*it;} // Commented-out debug line to print sorted elements.

  // Block Logic: Iterates through the sorted vector from the largest element downwards.
  //              Accumulates 'sum2' and increments 'cnt' until 'sum2' is greater
  //              than half of the total 'sum'.
  // Invariant: 'sum2' always holds the sum of the 'cnt' largest elements considered so far.
  for (vector<int>::iterator i = (ele.end() - 1); i >= ele.begin(); i--) {
    // cout<<*i; // Commented-out debug line to print current largest element.
    sum2 += (*i); // Add the current largest element to sum2.
    cnt++;        // Increment the count of elements taken.
    
    // Break condition: If the accumulated sum (sum2) exceeds half of the total sum.
    if (sum2 > (sum / 2)) {
      break;
    }
  }
  
  // Functional Utility: Prints the minimum count of largest elements needed.
  cout << cnt;
  
  return 0; // Indicates successful program termination.
}
