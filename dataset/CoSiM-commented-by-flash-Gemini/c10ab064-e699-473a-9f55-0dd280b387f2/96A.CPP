#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int i, j, flag = 0;
  char str[110];
  cin >> str;
  if (strlen(str) >= 7) {
    for (i = 0; i < strlen(str) - 6; i++) {
      if ((str[i] == str[i + 1]) && (str[i + 1] == str[i + 2]) &&
          (str[i + 2] == str[i + 3]) && (str[i + 3] == str[i + 4]) &&
          (str[i + 4] == str[i + 5]) && (str[i + 5] == str[i + 6])) {
        flag = 1;
        break;
      }
    }
  }
  if (flag == 1)
    cout << "YES" << endl;
  else
    cout << "NO" << endl;
  return 0;
}
