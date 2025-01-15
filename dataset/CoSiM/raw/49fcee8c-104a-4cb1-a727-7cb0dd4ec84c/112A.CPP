#include <cstring>
#include <iostream>
using namespace std;
main() {
  string str1, str2;
  cin >> str1 >> str2;
  for (int i = 0; i < str1.length(); i++) {
    str1[i] = tolower(str1[i]);
    str2[i] = tolower(str2[i]);
  }
  if (str1 < str2) {
    cout << "-1";
  } else if (str2 < str1) {
    cout << "1";
  } else if (str1 == str2) {
    cout << "0";
  }
}
