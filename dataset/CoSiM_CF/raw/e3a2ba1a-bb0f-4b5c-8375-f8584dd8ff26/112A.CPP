#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
using namespace std;

char replace(char x) {
  if (x >= 'A' && x <= 'Z')
    return x + 32;
  return x;
}

int main() {
  string str1 = " ", str2 = " ";
  int res = 2;

  cin >> str1 >> str2;
  for (int i = 0; i < str1.length(); i++) {
    res = int(replace(str1[i])) - int(replace(str2[i]));
    if (res > 0) {
      cout << "1" << endl;
      break;
    } else if (res < 0) {
      cout << "-1" << endl;
      break;
    }
  }
  if (res == 0) {
    cout << "0" << endl;
  }
}
