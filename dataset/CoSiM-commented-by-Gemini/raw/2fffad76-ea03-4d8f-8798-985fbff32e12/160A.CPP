/**
 * @file 160A.CPP
 * @brief Solves the "Twins" problem by finding the minimum number of coins for a majority sum.
 * @details This program determines the minimum number of coins one needs to take from a set
 * so that the sum of the taken coins is strictly greater than the sum of the remaining coins.
 *
 * @algorithm The solution employs a greedy strategy. The most effective approach is to always
 * take the largest available coins first. The code implements this by sorting all coins in
 * descending order and then accumulating them one by one until their sum exceeds the sum of
 * the rest. The sorting is performed by a custom selection sort implementation.
 *
 * @time_complexity O(N^2), where N is the number of coins. The dominant operation is the
 * `ordenaDecreciente` function, which is a selection sort algorithm with a quadratic time
 * complexity. A more optimal solution using `std::sort` would achieve O(N log N).
 * @space_complexity O(N) to store the values of the coins in an array.
 */
/* P160A by Gabriel Scotillo */
#include <bits/stdc++.h>

using namespace std;

/**
 * @brief Swaps two integer elements in an array.
 * @param a The array of integers.
 * @param j Index of the first element.
 * @param i Index of the second element.
 */
void swap(int a[], int j, int i) {
  int k = a[j];
  a[j] = a[i];
  a[i] = k;
}

/**
 * @brief Sorts an array of integers in descending order using selection sort.
 * @note This is an O(N^2) sorting algorithm, less efficient than `std::sort`.
 * @param a The array to be sorted.
 * @param N The number of elements in the array.
 */
void ordenaDecreciente(int a[], int N) {
  int i = 0;
  // Block Logic: Implements a selection sort. For each element `i`, it finds the
  // largest element in the unsorted part of the array (`j` to `N-1`) and swaps it with `a[i]`.
  while (i < N) {
    int j = i + 1;
    while (j < N) {
      if (a[j] >= a[i]) {
        swap(a, j, i);
      }
      j++;
    }
    i++;
  }
}

int main() {
  // res: The result, counting the minimum number of coins taken.
  int res = 0;
  int n;
  cin >> n;
  const int N = n;
  // sumTot: The total value of all coins.
  int sumTot = 0;
  // sumParcial: The running sum of the coins taken.
  int sumParcial = 0;
  int a[N];
  // Block Logic: Read all coin values and calculate their total sum.
  for (int i = 0; i < N; i++) {
    cin >> a[i];
    sumTot = sumTot + a[i];
  }

  // Sort the coins in descending order to enable the greedy approach.
  ordenaDecreciente(a, N);

  /**
   * Block Logic: Greedily take the largest coins until their sum is greater than the
   * sum of the remaining coins.
   * Invariant: `sumParcial` holds the sum of the `res` largest coins.
   */
  for (int i = 0; i < N; i++) {
    // Condition to check if our partial sum is already greater than the rest.
    if (sumParcial > sumTot - sumParcial) {
      break;
    } else {
      // Take the next largest coin.
      sumParcial += a[i];
      res++;
    }
  }

  cout << res << endl;
  return 0;
}
