#include <bits/stdc++.h>
using namespace std;

string str;
int nNumbers;

int main() {
  cin >> str;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '4' or str[i] == '7') {
      nNumbers++;
    }
  }

  if (nNumbers == 4 or nNumbers == 7) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
}
