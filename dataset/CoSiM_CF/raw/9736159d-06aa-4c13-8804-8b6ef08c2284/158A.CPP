/*
 * Link: https://codeforces.com/problemset/problem/158/A
 */

#include <iostream>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  int *scores = new int[n];
  for (int i = 0; i < n; i++) {
    cin >> scores[i];
  }
  int passed = 0, i = 0;
  while ((scores[i] >= scores[k - 1]) && scores[i] > 0 && i < n) {
    passed++;
    i++;
  }
  cout << passed << endl;
}