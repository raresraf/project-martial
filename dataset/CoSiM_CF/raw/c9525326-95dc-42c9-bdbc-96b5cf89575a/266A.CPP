#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  char r[51];
  int counter = 0;
  cin >> n;

  for (int i = 0; i < n; ++i) {
    cin >> r[i];
  }

  for (int i = 0; i < n - 1; ++i) {
    if (r[i] == r[i + 1]) {
      counter++;
    }
  }

  cout << counter;
  return 0;
}
