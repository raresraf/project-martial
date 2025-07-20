#include <bits/stdc++.h>
using namespace std;
int main() {
  int n;
  cin >> n;
  int count = 0;
  while (n--) {
    int sum = 0;
    int a[3];
    for (int i = 0; i < 3; i++) {
      cin >> a[i];
    }
    for (int i = 0; i < 3; i++) {
      sum += a[i];
    }
    if (sum >= 2) {
      count++;
    }
  }
  cout << count;
}
