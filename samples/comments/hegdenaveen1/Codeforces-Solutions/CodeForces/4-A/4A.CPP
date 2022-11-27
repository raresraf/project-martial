#include <bits/stdc++.h>
using namespace std;

int main() {

#ifndef ONLINE_JUDGE
  // for getting input from input.txt
  freopen("input.txt", "r", stdin);
  // for writing output to output.txt
  freopen("output.txt", "w", stdout);
#endif

  int a;
  cin >> a;
  if (a > 3 && a % 2 == 0)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}