#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n;
  cin >> n;
  int max = 0;

  vector<pair<int, int>> pairs(n);
  for (int i = 0; i < n; i++) {
    cin >> pairs[i].first;
    cin >> pairs[i].second;
  }

  int maxval = pairs[0].first;

  for (int i = 0; i < n; i++) {
    max -= pairs[i].first;
    max += pairs[i].second;
    if (maxval < max)
      maxval = max;
  }
  cout << max;
  return 0;
}