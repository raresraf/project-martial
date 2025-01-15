#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;

  vector<int> v;
  for (int i = 0; i < n; i++) {
    int val;
    cin >> val;
    v.push_back(val);
  }

  // k-th place finisher's score
  int score;
  score = v[k - 1];

  // the answer; the number of contestances who's score is >= k-th place
  // finisher's score.
  int nextRound = 0;
  for (int i = 0; i < n; i++) {
    if (v[i] >= score && v[i] != 0) {
      nextRound++;
    }
  }
  cout << nextRound << endl;
  return 0;
}