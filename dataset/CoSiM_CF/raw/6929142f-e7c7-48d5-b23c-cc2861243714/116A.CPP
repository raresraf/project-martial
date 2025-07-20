#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int total = 0, temp = 0;
  for (int i = 0; i < n; i++) {
    int a, b;
    cin >> a >> b;
    temp -= a;
    temp += b;

    total = max(temp, total);
  }

  cout << total;
}
