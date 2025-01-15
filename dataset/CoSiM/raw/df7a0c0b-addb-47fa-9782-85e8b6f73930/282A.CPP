#include <bits/stdc++.h>
using namespace std;
int main() {
  int t, sum = 0;
  string s;
  for (scanf("%d", &t); t--;) {
    cin >> s;
    if (s[1] == '+')
      sum++;
    else
      sum--;
  }
  cout << sum << endl;
  return 0;
}
