#include "string"
#include <iostream>
using namespace std;

int main(int argc, const char *argv[]) {
  string str, res;
  cin >> str;

  for (int i = 0; i < str.length(); i++) {
    if (str[i] == 'a' || str[i] == 'e' || str[i] == 'i' || str[i] == 'o' ||
        str[i] == 'u' || str[i] == 'y' || str[i] == 'A' || str[i] == 'E' ||
        str[i] == 'O' || str[i] == 'I' || str[i] == 'U' || str[i] == 'Y')
      continue;
    else {
      res += '.';
      res += towlower(str[i]);
    }
  }
  cout << res;

  return 0;
}
