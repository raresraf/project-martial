#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int counter = 0;
  for (int i = 0; i < n; i++) {
    int a, b, c;
    cin >> a >> b >> c;
    if (a + b + c > 1)
      counter++;
  }

  cout << counter;
}
