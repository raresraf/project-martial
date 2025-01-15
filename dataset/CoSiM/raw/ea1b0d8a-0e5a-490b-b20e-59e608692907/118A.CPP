#include <iostream>
#include <string>
using namespace std;
string str;
int main() {
  cin >> str;
  int n;
  n = str.size();
  for (int i = 0; i < n; i++) {
    if (!(str[i] == 'A' || str[i] == 'O' || str[i] == 'Y' || str[i] == 'E' ||
          str[i] == 'U' || str[i] == 'I' || str[i] == 'a' || str[i] == 'o' ||
          str[i] == 'y' || str[i] == 'e' || str[i] == 'u' || str[i] == 'i')) {
      cout << '.' << char(tolower(str[i]));
    }
  }

  return 0;
}
