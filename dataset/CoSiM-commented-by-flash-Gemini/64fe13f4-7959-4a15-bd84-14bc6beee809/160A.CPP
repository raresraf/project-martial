/**
 * @64fe13f4-7959-4a15-bd84-14bc6beee809/160A.CPP
 * @brief This program appears to solve a competitive programming problem involving
 * a "Buy X, get Y free" type of discount scenario for a set of items.
 * Specifically, it seems to implement a "Buy 'r' items, get the (r+1)-th item free"
 * logic, where 'r' is determined by the minimum of several input discount values.
 * The goal is to calculate the total cost after applying this discount to the most
 * expensive items.
 *
 * Algorithm: Greedy approach.
 * 1. Determine the effective discount quantity 'r' (buy 'r' get 1 free) by finding the minimum of `m` input values.
 * 2. Calculate the total cost of all 'n' items.
 * 3. Sort the items by cost in ascending order.
 * 4. Iterate from the most expensive items downwards. For every `r` items, the next cheapest item
 *    (the `(r+1)`-th from the group being considered from the end) is effectively free.
 *    The cost of this "free" item is subtracted from the total.
 * Time Complexity: O(N log N) due to sorting, plus O(N) for iteration. Overall O(N log N).
 * Space Complexity: O(N) for storing item costs.
 */
/*
 *	Copyright (C) Ds ds303077135@gmail.com
 *	File Name     : A.cpp
 *	Creation Time : 2013/01/08 18:34:10
 *	Environment   : OS X 10.8
 */
#include <algorithm> // For std::sort and std::min.
#include <cmath>     // For mathematical functions (unused in this specific logic).
#include <cstdio>    // For C-style input/output (scanf).
#include <cstdlib>   // For general utilities (unused in this specific logic).
#include <cstring>   // For C-style string functions (unused in this specific logic).
#include <iostream>  // For standard input/output (cout, endl).
#include <map>       // For std::map (unused in this specific logic).
#include <set>       // For std::set (unused in this specific logic).

/**
 * @brief Macro for a for-loop that iterates from 1 up to (and including) n.
 * @param i Loop variable.
 * @param n Ending value (inclusive).
 */
#define rep(i, n) for (int i = 1; i <= n; ++i)
using namespace std;

int m, n, q, r = 100001; // @brief 'm': number of discount settings; 'n': number of items; 'q': temporary for reading discount; 'r': effective 'buy N' quantity, initialized to a large value.
int a[100010];           // @brief Array 'a' to store the costs of up to 100,000 items.
long long tot;           // @brief 'tot': accumulates the total cost of items after discounts.

int main() {
  // Block Logic: Reads the number of discount settings ('m') and then 'm' individual discount quantities ('q').
  // It finds the minimum 'q' value and stores it in 'r', which represents the 'buy N' part of the discount.
  scanf("%d", &m);
  rep(i, m) scanf("%d", &q), r = min(r, q); // Iterates 'm' times, reading 'q' and updating 'r' with the minimum.
  
  // Block Logic: Reads the total number of items ('n') and then 'n' item costs into array 'a'.
  // It also calculates the initial total cost ('tot') of all items without any discounts.
  scanf("%d", &n);
  rep(i, n) scanf("%d", &a[i]), tot += a[i]; // Iterates 'n' times, reading each item cost and adding to 'tot'.
  
  // Block Logic: Sorts the array 'a' (containing item costs) in ascending order.
  // This is crucial for the greedy strategy, as applying discounts to the cheapest
  // eligible items maximizes overall savings.
  sort(a + 1, a + 1 + n); // Sorts elements from a[1] to a[n].
  
  int pos = n; // 'pos' acts as a pointer to the current most expensive item being considered (1-indexed).
  
  /**
   * Block Logic: Iterates from the most expensive items downwards, applying the "buy 'r' get 1 free" discount.
   * Invariant: `tot` accurately reflects the sum after applying discounts to eligible items.
   */
  while (pos >= 1) { // Continues as long as there are items left to consider.
    // Pre-condition: Checks if there are enough items remaining (`pos`) to form a group for discount.
    // If `pos` items are left, and we need to buy `r` to get one free, then `pos > r` means a free item can be identified.
    if (pos > r) { 
      // Inline: Subtracts the cost of the `(r+1)`-th item from the end of the current `pos` items.
      // This is the item that effectively becomes "free".
      tot -= a[pos - r]; 
      
      // Block Logic: Advances 'pos' to skip the items just processed and the "free" item.
      // After processing `r` items and getting 1 free, the next group starts after `r+1` items.
      if (pos - r > 1) // If there are at least two items before the "free" item (in terms of index).
        tot -= a[pos - r - 1], pos = pos - r - 2; // Subtracts another item, and skips past two items.
                                                 // This part of the logic seems specific to a particular problem variant.
      else // If only one item before the "free" item.
        pos = pos - r - 1; // Skips past one item.
    } else {
      break; // No more items to apply the discount to.
    }
  }
  cout << tot << endl; // Prints the final total cost after all applicable discounts, followed by a newline.
  return 0; // Indicates successful program termination.
}
