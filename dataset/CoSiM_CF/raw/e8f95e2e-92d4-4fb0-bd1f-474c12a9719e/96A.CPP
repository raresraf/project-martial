#include <iostream>

using namespace std;

int main() {
  string str;
  int one = 0, zero = 0, flag = 0;
  cin >> str;
  for (int i = 0; i < str.length(); i++) {
    if (str[i] == '1') {
      one++;
      zero = 0;
    } else {
      zero++;
      one = 0;
    }
    if (zero == 7 || one == 7) {
      flag = 1;
      break;
    }
  }
  if (flag)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;

  return 0;
}
