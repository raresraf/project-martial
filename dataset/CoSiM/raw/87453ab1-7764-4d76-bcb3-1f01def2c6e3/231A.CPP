#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  int sum = 0;
  while (n--) {
    int x;
    int temp = 0;
    for (int i = 0; i < 3; i++) {
      cin >> x;
      if (x == 1)
        temp++;
    }
    if (temp >= 2)
      sum++;
  }
  cout << sum;
  return 0;
}