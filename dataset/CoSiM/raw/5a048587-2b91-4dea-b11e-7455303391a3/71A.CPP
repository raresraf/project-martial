#include <iostream>
#include <string.h>
using namespace std;

int main() {
  int n, i;
  char str[1000];

  cin >> n;

  if (n >= 1) {
    while (n--) {
      cin >> str;
      i = strlen(str);
      if ((i >= 1) && (i <= 10))
        cout << str << endl;
      else if (i <= 100)
        cout << str[0] << (i - 2) << str[i - 1] << endl;
    }
  }

  return 0;
}
