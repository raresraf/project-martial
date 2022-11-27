/*
Problem description:

https://codeforces.com/problemset/problem/4/A
*/

#include <bits/stdc++.h>

using namespace std;

int main() {
  int w;
  cin >> w;
  string answer = w % 2 == 0 && w > 2 ? "YES" : "NO";
  cout << answer;
  return 0;
}
