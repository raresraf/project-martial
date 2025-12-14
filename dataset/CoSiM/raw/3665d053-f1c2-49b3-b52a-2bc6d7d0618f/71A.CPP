#include <iostream>
#include <string.h>

using namespace std;

int main() {

  int n, len;
  char str[100][101];

  cin >> n;

  for (int i = 0; i < n; ++i) {
    cin >> str[i];
  }

  for (int i = 0; i < n; ++i) {

    len = strlen(str[i]) - 2;

    if (len <= 8) {
      cout << str[i] << endl;

    } else {
      cout << str[i][0] << len << str[i][len + 1] << endl;
    }
  }

  return 0;
}