#include <bits/stdc++.h>
using namespace std;

int main() {
  int stops;
  cin >> stops;
  int maximum = -1;
  int cur = 0;
  while (stops--) {
    int ups, downs;
    cin >> downs >> ups;
    cur = cur + ups - downs;
    maximum = max(maximum, cur);
  }
  cout << maximum;
  return 0;
}