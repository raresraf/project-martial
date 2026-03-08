/**
 * @file 339A_2.CPP
 * @brief Reorders numbers in a mathematical sum string (e.g., "1+3+2") into non-decreasing order ("1+2+3").
 *
 * Problem Name: Helpful Maths (Codeforces 339A)
 * Algorithm: Extracts single-digit numbers from the input string, sorts them using a custom Quicksort implementation,
 *            and then reconstructs and prints the sum string.
 * Time Complexity: Average O(N log N) dominated by Quicksort, where N is the count of numbers in the string.
 *                  Worst case O(N^2) for Quicksort. Parsing and printing are O(L) where L is string length.
 * Space Complexity: O(N) for storing the extracted numbers. Quicksort uses additional stack space (O(log N) to O(N)).
 */
// By archit-dev, contest: Codeforces Round #197 (Div. 2), problem: (A) Helpful
// Maths, Accepted, #, Copy
#include <algorithm> // Generally included for std::sort, but here a custom quicksort is used.
#include <bits/stdc++.h> // Comprehensive header, often used in competitive programming.
#include <iostream>  // Required for input/output operations (cin, cout).
#include <string>    // Required for std::string manipulations.
#define ll long long // Functional Utility: Defines 'll' as a shorthand for 'long long'.
using namespace std;

/**
 * @brief Declaration of the quick_sort function.
 * @param a The array to be sorted.
 * @param l The lower index of the current partition.
 * @param u The upper index of the current partition.
 */
void quick_sort(int a[], int l, int u);
/**
 * @brief Declaration of the partition function used by quick_sort.
 * @param a The array to be partitioned.
 * @param l The lower index of the current partition.
 * @param u The upper index of the current partition.
 * @return The pivot index after partitioning.
 */
int partition(int a[], int l, int u);

/**
 * @brief Main function to handle input, process the string, and print the sorted sum.
 * @pre Input consists of a single string representing a sum of single-digit numbers (e.g., "1+3+2").
 * @post Prints the sum string with numbers reordered in non-decreasing order (e.g., "1+2+3").
 */
int main() {
  string s; // Variable to store the input sum string.
  // Functional Utility: Reads the input sum string from standard input.
  cin >> s;

  // Calculates the logical size needed for the array of numbers.
  // If string length is odd (e.g., "1"), size is 1. If even (e.g., "1+2"), numbers are 2, string length 3.
  // The numbers are at s[0], s[2], s[4]...
  int size = s.size() / 2;
  if (s.size() % 2 != 0) {
    size += 1;
  }
  int a[size] = {0}; // Array to store the extracted single-digit numbers.
  int k = 0;         // Index for populating the 'a' array.

  /**
   * Block Logic: Extracts single-digit numbers from the input string.
   * Invariant: It iterates through the string, taking characters at even indices, converts them from char to int,
   *            and stores them in array 'a'.
   */
  for (int i = 0; i < s.size(); i += 2) {
    a[k] = s[i] - 48; // Converts character digit to integer (ASCII '0' is 48).
    k++;              // Moves to the next position in array 'a'.
  }

  // Functional Utility: Sorts the extracted numbers in array 'a' using the custom quicksort.
  quick_sort(a, 0, size - 1); // Quicksort typically uses (size-1) as upper bound for 0-indexed array.

  /**
   * Block Logic: Prints the sorted numbers, reinserting '+' signs between them.
   * Invariant: The last number is printed with a newline. Other numbers are followed by a '+'.
   */
  for (int i = 0; i < size; i++) {
    if (i == size - 1) { // Checks if it's the last number.
      cout << a[i] << endl;
    } else {
      cout << a[i] << "+";
    }
  }
  // Indicates successful program execution.
  return 0;
}

/**
 * @brief Implements the recursive quicksort algorithm.
 * @param a The array to be sorted.
 * @param l The lower index of the current partition.
 * @param u The upper index of the current partition.
 * @pre `a` is a valid integer array, `l` and `u` define a valid sub-array range.
 * @post The sub-array `a[l...u]` is sorted in non-decreasing order.
 */
void quick_sort(int a[], int l, int u) {
  int j; // Pivot index returned by partition.
  /**
   * Block Logic: Base condition for recursion: if the lower bound is less than the upper bound,
   *            meaning there is more than one element to sort.
   * Invariant: Recursively sorts sub-arrays until all partitions are of size 0 or 1.
   */
  if (l < u) {
    j = partition(a, l, u); // Partitions the array and gets the pivot index.
    quick_sort(a, l, j - 1); // Recursively sorts the left sub-array.
    quick_sort(a, j + 1, u); // Recursively sorts the right sub-array.
  }
}

/**
 * @brief Partitions an array segment around a pivot element.
 * @param a The array segment to be partitioned.
 * @param l The lower index of the current segment.
 * @param u The upper index of the current segment.
 * @return The final index of the pivot element.
 * @pre `a` is a valid integer array, `l` and `u` define a valid segment.
 * @post Elements smaller than the pivot are moved to its left, and larger elements to its right.
 */
int partition(int a[], int l, int u) {
  int v, i, j, temp; // 'v': pivot element, 'i', 'j': pointers for traversal, 'temp': for swapping.

  v = a[l]; // Chooses the first element as the pivot.
  i = l;    // Pointer starting from the left.
  j = u + 1; // Pointer starting from the right (one position beyond the end).

  /**
   * Block Logic: Rearranges elements such that elements less than the pivot are on its left,
   *            and elements greater than the pivot are on its right.
   * Invariant: `i` moves right past elements smaller than pivot. `j` moves left past elements larger than pivot.
   *            Elements are swapped when `a[i]` is greater than pivot and `a[j]` is smaller than pivot.
   */
  do {
    do
      i++;
    while (a[i] < v && i <= u); // Finds an element greater than or equal to pivot from the left.

    do
      j--;
    while (v < a[j]); // Finds an element smaller than or equal to pivot from the right.

    /**
     * Block Logic: Swaps elements if 'i' and 'j' haven't crossed.
     * Pre-condition: `i` and `j` point to elements that are out of place relative to the pivot.
     * Invariant: `a[i]` and `a[j]` are swapped, moving them closer to their correct partitioned positions.
     */
    if (i < j) {
      temp = a[i];
      a[i] = a[j];
      a[j] = temp;
    }
  } while (i < j); // Continues until pointers 'i' and 'j' cross.

  // Functional Utility: Places the pivot element in its correct sorted position.
  a[l] = a[j]; // Swaps the original pivot with the element at 'j'.
  a[j] = v;    // Places the pivot into its final position.

  return (j); // Returns the final pivot index.
}
