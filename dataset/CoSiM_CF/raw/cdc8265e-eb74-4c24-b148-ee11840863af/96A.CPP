#include <bits/stdc++.h>
using namespace std;
int main() {
  char ch[102];
  int a = 0, b = 0;
  cin >> ch;
  for (int i = 0; i < strlen(ch); i++) {

    if (ch[i] == '0') {
      a++;
      if (b < 7)
        b = 0;
    } else {
      b++;
      if (a < 7)
        a = 0;
    }
  }
  if (a >= 7 || b >= 7)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
}
