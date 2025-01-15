//
// Created by Conor on 5/27/2020.
//

#include <iostream>
#include <iterator>
#include <vector>
using namespace std;

int main() {
  int n, k;
  cin >> n >> k;
  vector<int> scores;

  int v;
  for (int i = 0; i < n; ++i) {
    cin >> v;
    scores.push_back(v);
  }

  int count;
  count = 0;
  for (int j = 0; j < scores.size(); ++j) {
    if (scores.at(j) < scores.at(k - 1) || scores.at(j) <= 0) {
      cout << count;
      return 0;
    }
    count++;
  }

  cout << count << endl;

  return 0;
}