#include <bits/stdc++.h>
using namespace std;
main() {
  int n, s = 0;
  cin >> n;
  for (int i = 2; i < n; i++)
    s = s + i * (i + 1);
  cout << s << endl;
}
