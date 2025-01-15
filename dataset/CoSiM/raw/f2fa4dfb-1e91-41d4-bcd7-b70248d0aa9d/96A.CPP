#include <iostream>
using namespace std;

int main() {
  string s;
  cin >> s;
  int l = s.length();

  int flag = s[0];
  int cnt = 1;
  int p = 0;

  for (int i = 1; i < l; i++) {
    if (s[i] == flag) {
      cnt++;
      // cout<<cnt<<endl;
    }

    else {
      flag = s[i];
      cnt = 1;
    }

    if (cnt == 7) {
      cout << "YES" << endl;
      p = 1;
      break;
    }
  }

  if (p == 0)
    cout << "NO" << endl;

  return 0;
}
