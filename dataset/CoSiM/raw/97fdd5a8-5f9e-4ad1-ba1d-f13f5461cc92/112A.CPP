#include <iostream>
using namespace std;

main() {
  string str, str1;
  cin >> str >> str1;
  int n = str.length();
  int flag;
  for (int i = 0; i < n; i++) {
    str[i] = tolower(str[i]);
    str1[i] = tolower(str1[i]);
    if (str[i] > str1[i]) {
      flag = 1;
      break;
    } else if (str[i] < str1[i]) {
      flag = -1;
      break;
    } else
      flag = 0;
  }
  if (flag == 1)
    cout << "1" << endl;
  else if (flag == -1)
    cout << "-1" << endl;
  else
    cout << "0" << endl;
}
