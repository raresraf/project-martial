#include <iostream>
using namespace std;

int main() {
  int n;
  string str;
  cin >> n;
  while (n--) {
    cin >> str;
    if (str.length() > 10)
      cout << str[0] << str.length() - 2 << str[str.length() - 1] << endl;
    else
      cout << str;
  }
  return 0;
}
