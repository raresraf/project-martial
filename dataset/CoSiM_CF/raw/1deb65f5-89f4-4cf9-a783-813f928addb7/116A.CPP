#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, a, b, max_value, current;
  max_value = current = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> a >> b;
    current += b;
    current -= a;
    if (current > max_value)
      max_value = current;
  }
  cout << max_value;
  return 0;
}
