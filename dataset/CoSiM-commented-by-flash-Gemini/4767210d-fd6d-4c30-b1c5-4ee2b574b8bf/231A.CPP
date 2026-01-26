// https://codeforces.com/problemset/problem/231/A
// 231A Team

#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, count = 0;
  cin >> n;
  while (n--) {
    int a, b, c;
    cin >> a >> b >> c;
    if (a + b + c >= 2)
      count++;
  }
  cout << count;
}
