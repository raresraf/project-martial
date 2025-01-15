#include "bits/stdc++.h"
using namespace std;

bool compare(int a, int b) { return a > b; }

int main(int argc, char const *argv[]) {
  int n;
  cin >> n;

  std::vector<int> v(n);

  std::vector<int> sum(n + 1, 0);

  for (int i = 0; i < n; i++) {
    cin >> v[i];
  }

  sort(v.begin(), v.end(), compare);

  for (int i = 0; i < n; i++) {
    sum[i + 1] = sum[i] + v[i];
  }

  int i;
  for (i = 1; i < n + 1; i++) {
    if (sum[i] > sum[n] - sum[i])
      break;
  }

  cout << i;
  return 0;
}
