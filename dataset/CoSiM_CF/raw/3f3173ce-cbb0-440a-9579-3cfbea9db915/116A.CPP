#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int m = INT_MIN;
  int num[n * 2];
  for (int i = 0; i < n * 2; i++) {
    cin >> num[i];
  }
  for (int i = 0; i < n * 2; i++) {
    int k = 0;
    if (i % 2 == 0) {
      num[i + 1] = num[i] + num[i + 1];
    } else {
      num[i + 1] = num[i] - num[i + 1];
    }
    if (num[i + 1] > m)
      m = num[i + 1];
  }
  cout << m << endl;

  return 0;
}
