#include <bits/stdc++.h>
using namespace std;

int main() {
  int n, length, tmp;
  string str, strfirst, str_num, strlast, str_result;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> str;
    length = str.size();
    if (length > 10) {
      strfirst = str[0];
      tmp = length - 2;

      stringstream convert;
      convert << tmp;
      str_num = convert.str();

      strlast = str[length - 1];
      str_result = strfirst + str_num + strlast;
      cout << str_result << endl;
    } else {
      cout << str << endl;
    }
  }
  return 0;
}
