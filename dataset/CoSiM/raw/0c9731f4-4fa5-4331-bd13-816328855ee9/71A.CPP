#include <iostream>
#include <stdlib.h>
#include <string.h>
using namespace std;

int main() {

  int n;
  string str;
  // string tmp = "";
  string str_ar[10000];

  cin >> n;

  if (n >= 1 && n <= 100) {
    int i = 0;
    while (i <= n) {
      getline(cin, str);
      if (str.length() > 10) {
        str_ar[i] += str[0];
        str_ar[i].append(to_string(str.length() - 2));
        str_ar[i] += str[str.length() - 1];
      } else {
        str_ar[i] = str;
      }
      i++;
    }
    i = 0;
    while (i <= n) {
      cout << str_ar[i] << endl;
      i++;
    }
  }

  else {
    exit(0);
  }

  return 0;
}
