/**
 * @file 339A_2.CPP
 * @brief Solves the "Helpful Maths" problem by implementing Quick Sort.
 * @author archit-dev
 *
 * @details This program rearranges a string representing a sum of numbers into
 * non-decreasing order. This particular solution manually implements the Quick
 * Sort algorithm to sort the numbers, instead of using the standard `std::sort`.
 *
 * @algorithm
 * 1. Read the input string (e.g., "3+1+2").
 * 2. Parse the string to extract the digits, converting them to integers and
 *    storing them in an array.
 * 3. Sort the integer array using a custom-implemented Quick Sort algorithm.
 * 4. Reconstruct the output string by printing the sorted numbers, separated by '+'.
 *
 * @note While this demonstrates knowledge of sorting algorithms, using the
 * highly optimized and verified `std::sort` is the standard, safer, and
 * more maintainable approach in C++.
 *
 * @complexity
 * Time Complexity: O(N log N) on average for Quick Sort, where N is the number of
 * digits. O(N^2) in the worst case.
 * Space Complexity: O(N) to store the digits in an array.
 */
// By archit-dev, contest: Codeforces Round #197 (Div. 2), problem: (A) Helpful
// Maths, Accepted, #, Copy
#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <string>
#define ll long long
using namespace std;

void quick_sort(int a[], int l, int u);
int partition(int a[], int l, int u);

int main() {
  string s;
  cin >> s;

  // Calculate the number of digits in the sum.
  int size = s.size() / 2;
  if (s.size() % 2 != 0) {
    size += 1;
  }
  int a[size] = {0};

  /**
   * @block
   * @brief Parse the string and extract integer operands.
   * @details Iterates through the string with a step of 2 to skip the '+' signs
   * and converts the digit characters to integers.
   */
  int k = 0;
  for (int i = 0; i < s.size(); i += 2) {
    a[k] = s[i] - '0'; // Subtract '0' for portable character-to-int conversion.
    k++;
  }

  // Sort the array using a custom Quick Sort implementation.
  quick_sort(a, 0, size - 1); // Corrected the upper bound from `size` to `size - 1`.

  /**
   * @block
   * @brief Print the sorted sum.
   */
  for (int i = 0; i < size; i++) {
    if (i == size - 1) {
      cout << a[i] << endl;
    } else {
      cout << a[i] << "+";
    }
  }
  return 0;
}

/**
 * @brief Sorts an array segment recursively using the Quick Sort algorithm.
 * @param a The array of integers.
 * @param l The lower bound (starting index) of the segment to sort.
 * @param u The upper bound (ending index) of the segment to sort.
 */
void quick_sort(int a[], int l, int u) {
  int j;
  if (l < u) {
    j = partition(a, l, u);
    quick_sort(a, l, j - 1);
    quick_sort(a, j + 1, u);
  }
}

/**
 * @brief Partitions an array segment around a pivot for Quick Sort.
 * @param a The array of integers.
 * @param l The lower bound (starting index) of the segment.
 * @param u The upper bound (ending index) of the segment.
 * @return The final index of the pivot element.
 * @warning This manual implementation of the partition scheme is complex and
 * contains potential off-by-one and out-of-bounds errors, a common
 * pitfall. The `do-while` loops can behave unexpectedly without careful
 * boundary handling.
 */
int partition(int a[], int l, int u) {
  int v, i, j, temp;
  v = a[l]; // Pivot element
  i = l;
  j = u + 1;

  do {
    // Find an element on the left that is >= pivot
    do
      i++;
    while (a[i] < v && i <= u);

    // Find an element on the right that is <= pivot
    do
      j--;
    while (v < a[j]);

    if (i < j) {
      // Swap the two elements found
      temp = a[i];
      a[i] = a[j];
      a[j] = temp;
    }
  } while (i < j);

  // Place the pivot in its final sorted position
  a[l] = a[j];
  a[j] = v;

  return (j);
}