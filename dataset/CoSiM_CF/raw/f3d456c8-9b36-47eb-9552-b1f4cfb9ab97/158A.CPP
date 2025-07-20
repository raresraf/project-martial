#include <bits/stdc++.h>

using namespace std;

bool wayToSort(int i, int j) { return i > j; }

int main() {
  int n, k;

  cin >> n >> k;

  int i, all_scores[n];

  for (i = 0; i < n; i++) {
    cin >> all_scores[i];
  }
  sort(all_scores, all_scores + n, wayToSort);

  int finalists = 0;
  int minimum_score = -1;
  for (i = 0; i < n; i++) {
    if (finalists >= k && all_scores[i] < minimum_score) {
      break;
    }
    if (all_scores[i] > 0) {
      finalists++;
      minimum_score = all_scores[i];
    }
  }

  cout << finalists << endl;
}

// g++ -std=c++11 -O2 -Wall 158A.cpp -o out