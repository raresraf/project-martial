//#include <bits/stdc++.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {

  int n, k, s, res = 0;

  cin >> n >> k;
  // construct vector
  vector<int> scores;

  // Initialize vector with scores
  for (int i = 0; i < n; i++) {
    cin >> s;
    scores.push_back(s);
  }

  for (int j = 0; j < n; j++) {
    if (scores[j] >= scores[k - 1] && !(scores[j] == 0))
      res++;
  }
  cout << res << endl;

  return 0;
}
