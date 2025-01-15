#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, d;
  int result = 0;
  int count = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    count = 0;
    for (int i = 0; i < 3; i++) {
      cin >> d;
      count += d;
    }
    if (count > 1) {
      result++;
    }
  }
  cout << result;
  return 0;
}
