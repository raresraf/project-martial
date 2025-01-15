#include <bits/stdc++.h>

using namespace std;

void read() {
  int counterz = 0, countero = 0;
  int flag;
  string s;
  cin >> s;

  if (s[0] == '0')
    flag = 0;
  else if (s[0] == '1')
    flag = 1;

  for (int i = 0; i < s.size(); ++i) {
    if (s[i] == '1' && flag == 1) {
      countero++;
    } else if (s[i] == '1' && flag == 0) {
      flag = 1;
      countero = 1;
      counterz = 0;
    } else if (s[i] == '0' && flag == 0) {
      counterz++;
    } else if (s[i] == '0' && flag == 1) {
      flag = 0;
      counterz = 1;
      countero = 0;
    }

    if (countero >= 7 || counterz >= 7) {
      cout << "YES" << endl;
      return;
    }
  }

  cout << "NO" << endl;
  return;
}

int main() {
  read();
  return 0;
}