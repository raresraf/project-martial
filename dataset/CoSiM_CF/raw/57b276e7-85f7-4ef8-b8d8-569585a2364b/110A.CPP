#include <bits/stdc++.h>
using namespace std;
int main() {
  string m;
  int c = 0, d = 0;
  cin >> m;
  for (int i = 0; m[i] != '\0'; i++) {
    if (m[i] == '4')
      c++;
    else if (m[i] == '7')
      d++;
  }

  if ((c + d) == 4 || (c + d) == 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}