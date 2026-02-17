/**
 * @file 546A.CPP
 * @brief Solution for a competitive programming problem about counting remaining book chapters.
 *
 * @details This program solves a problem where we are given the page ranges for
 * `N` chapters in a book and the current page `K` we are reading. The goal is to
 * determine how many chapters are not yet fully read. A chapter is considered
 * finished only if the current page `K` is past the chapter's last page.
 *
 * @section algorithm
 * Algorithm:
 * The approach is a straightforward linear scan through all the chapters.
 * 1. Read the total number of chapters, `N`.
 * 2. Read the start and end page for each of the `N` chapters. Only the end page
 *    is relevant for the logic.
 * 3. Read the current page number, `K`.
 * 4. Initialize a counter for "not yet read" chapters to `N`.
 * 5. Iterate through each chapter.
 * 6. For each chapter, check if the current page `K` is greater than the
 *    chapter's end page. If it is, the chapter is finished, so decrement the
 *    counter.
 * 7. After checking all chapters, print the final count of chapters that are
 *    not yet finished.
 *
 * @section complexity
 * Time Complexity: O(N), because the code iterates through the `N` chapters exactly once.
 * Space Complexity: O(N), to store the page ranges for the `N` chapters.
 */
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;

int N, K;
// Array to store the start (first) and end (second) pages of each chapter.
pair<int, int> chapters[100010];

int main() {
  cin >> N;
  // Block Logic: Read the page range for each of the N chapters.
  // The first element of the pair (start page) is read but not used in the logic.
  for (int i = 0; i < N; i++)
    cin >> chapters[i].first >> chapters[i].second;
  cin >> K;

  // 'nryet' stands for "number not yet read". Initialize with the total number of chapters.
  int nryet = N;

  // Block Logic: Iterate through chapters to find how many are already finished.
  // Pre-condition: 'chapters' array is filled and 'K' is the current page.
  for (int i = 0; i < N; i++) {
    // A chapter is considered finished if the reader is past its last page.
    if (K > chapters[i].second) {
      nryet--; // Decrement the count of unread chapters.
    }
  }

  // Output the final number of chapters that are not yet fully read.
  cout << nryet << endl;
  return 0;
}