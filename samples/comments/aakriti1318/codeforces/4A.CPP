// 4A watermelon
// https://codeforces.com/problemset/problem/4/A

#include <bits/stdc++.h>
using namespace std;

int main() {
  int n;
  cin >> n;
  if (n % 2 == 0) {
    if (n <= 2)
      cout << "NO";
    else
      cout << "YES";
  } else
    cout << "NO";
  return 0;
}