#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, a, b;
  cin >> n;
  int max = 0;
  int sum = 0;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    sum -= a;
    sum += b;
    if (sum > max)
      max = sum;
  }
  cout << max << endl;
  return 0;
}
