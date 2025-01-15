#include <cstring>
#include <iostream>

using namespace std;
int main() {
  int i, flag = 0;
  char str[100], str2[100];
  cin >> str;
  cin >> str2;
  for (i = 0; i < strlen(str); i++) {
    if ((str[i] <= 'Z') && (str[i] >= 'A'))
      str[i] = (char)((int)str[i] + 32);
    if ((str2[i] <= 'Z') && (str2[i] >= 'A'))
      str2[i] = (char)((int)str2[i] + 32);
    if (str[i] > str2[i]) {
      cout << "1" << '\n';
      flag = 1;
      break;
    } else if (str[i] < str2[i]) {
      cout << "-1" << '\n';
      flag = 1;
      break;
    }
  }
  if (flag == 0)
    cout << "0" << '\n';
  return 0;
}
