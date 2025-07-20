#include <bits/stdc++.h>
using namespace std;

int main(void) {
  char p[102];
  int z = 0;
  int o = 0;
  int i;
  cin >> p;
  for (i = 0; p[i] != '\0'; i++) {
    if (p[i] == '0') {
      z++;
      o = 0;
    } else if (p[i] == '1') {
      o++;
      z = 0;
    }
    if (z == 7 || o == 7)
      break;
  }
  if (z == 7 || o == 7)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}
