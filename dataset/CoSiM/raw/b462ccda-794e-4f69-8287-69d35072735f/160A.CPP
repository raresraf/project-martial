#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, sum = 0, newsum = 0, c = 0;
  cin >> n;
  vector<int> seq(n);

  for (int i = 0; i < n; ++i) {
    cin >> seq[i];
    sum += seq[i];
  }
  sort(seq.begin(), seq.end());

  for (int i = n - 1; i >= 0; --i) {
    sum -= seq[i];
    newsum += seq[i];
    // cout << "seq[i]: " << seq[i]
    // << "sum: " << sum << "newsum: " << newsum << endl;
    c++;
    if (newsum > sum)
      break;
  }

  cout << c << endl;

  return 0;
}

// https://codeforces.com/problemset/problem/160/A