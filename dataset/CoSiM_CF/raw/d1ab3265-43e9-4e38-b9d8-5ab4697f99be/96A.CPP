#include <iostream>
#include <string>
using namespace std;

int main() {

  int cnt = 0, cnt1 = 0;
  string str, str1 = "";

  cin >> str;
  int x = str.length();
  for (int i = 0; i < x; i++) {
    if (str[i] == '0') {
      cnt1 = 0;
      cnt++;
    }

    else if (str[i] == '1') {
      cnt = 0;
      cnt1++;
    }

    if (cnt1 == 7 || cnt == 7) {
      cout << "YES" << endl;
      return 0;
    }
  }
  cout << "NO" << endl;
  return 0;
}
